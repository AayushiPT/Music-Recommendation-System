from flask import Flask, request, render_template, url_for, session, redirect
from functools import lru_cache
from Levenshtein import distance as levenshtein_distance
import numpy as np
import pandas as pd
import pickle
import random
import heapq

# Load dataset and similarity matrix
df = pickle.load(open('C:/Users/ccnd1/Desktop/Project_Sem_7/df.pkl', 'rb'))
similarity = pickle.load(open('C:/Users/ccnd1/Desktop/Project_Sem_7/similarity.pkl', 'rb'))

# Precompute heuristic cache for faster A* algorithm
heuristic_cache = {}

def precompute_heuristics():
    global heuristic_cache
    genre_indices = {genre: df[df['Genre'] == genre].index.tolist() for genre in df['Genre'].unique()}
    for genre1, indices1 in genre_indices.items():
        for genre2, indices2 in genre_indices.items():
            if genre1 != genre2:
                min_distance = min(1 - similarity[i][j] for i in indices1 for j in indices2)
                heuristic_cache[(genre1, genre2)] = min_distance

# Call this function once during initialization
precompute_heuristics()

def heuristic(current_index, target_indices):
    target_genre = df.iloc[target_indices[0]]['Genre']
    current_genre = df.iloc[current_index]['Genre']
    return heuristic_cache.get((current_genre, target_genre), float('inf'))

# Recommendation function using similarity
def recommendation(genre, max_recommendations=20):
    if genre not in df['Genre'].unique():
        return []
    idx = df[df['Genre'] == genre].index[0]
    distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])
    recommended_songs = []
    for i in distances[1:max_recommendations + 1]:
        recommended_songs.append({
            "Song-Name": df.iloc[i[0]]['Song-Name'],
            "Singer/Artists": df.iloc[i[0]]['Singer/Artists'],
            "Genre": df.iloc[i[0]]['Genre'],
            "Album/Movie": df.iloc[i[0]]['Album/Movie'],
            "User-Rating": df.iloc[i[0]]['User-Rating']
        })
    return recommended_songs

# A* algorithm for recommendation
SIMILARITY_THRESHOLD = 0.2

def a_star_recommendation(start_genre, target_genre, max_recommendations=30):
    start_indices = df[df['Genre'] == start_genre].index.tolist()
    target_indices = df[df['Genre'] == target_genre].index.tolist()

    if not start_indices or not target_indices:
        return []

    recommendations = []
    visited = set()

    # Step 1: Recommend songs from the start genre
    start_pq = []
    for index in start_indices:
        heapq.heappush(start_pq, (0, index, [index]))  # Initialize with heuristic 0

    start_genre_recommendations = []
    while start_pq and len(start_genre_recommendations) < max_recommendations // 2:
        _, current_index, _ = heapq.heappop(start_pq)
        if current_index in visited:
            continue
        visited.add(current_index)

        start_genre_recommendations.append({
            "Song-Name": df.iloc[current_index]['Song-Name'],
            "Singer/Artists": df.iloc[current_index]['Singer/Artists'],
            "Genre": df.iloc[current_index]['Genre'],
            "Album/Movie": df.iloc[current_index]['Album/Movie'],
            "User-Rating": df.iloc[current_index]['User-Rating']
        })

    # Step 2: Recommend songs from the target genre
    target_pq = []
    for index in target_indices:
        heapq.heappush(target_pq, (0, index, [index]))  # Initialize with heuristic 0

    target_genre_recommendations = []
    while target_pq and len(target_genre_recommendations) < max_recommendations // 2:
        _, current_index, _ = heapq.heappop(target_pq)
        if current_index in visited:
            continue
        visited.add(current_index)

        target_genre_recommendations.append({
            "Song-Name": df.iloc[current_index]['Song-Name'],
            "Singer/Artists": df.iloc[current_index]['Singer/Artists'],
            "Genre": df.iloc[current_index]['Genre'],
            "Album/Movie": df.iloc[current_index]['Album/Movie'],
            "User-Rating": df.iloc[current_index]['User-Rating']
        })

    # Combine results
    recommendations.extend(start_genre_recommendations)
    recommendations.extend(target_genre_recommendations)

    return recommendations

# Flask App
app = Flask(__name__)
app.secret_key = 'your_secret_key'

@app.route('/')
def index():
    genres = sorted(df['Genre'].unique()) if 'Genre' in df else []
    return render_template('MainR.html', genres=genres[:20])

@app.route('/recom', methods=['POST'])
def mysong():
    user_genre = request.form['genre']
    sort_order = request.form.get('sort_order', None)
    songs = recommendation(user_genre, max_recommendations=20)
    if not sort_order:
        random.shuffle(songs)
    elif sort_order == 'Ascending':
        songs = sorted(songs, key=lambda x: x['User-Rating'])
    elif sort_order == 'Descending':
        songs = sorted(songs, key=lambda x: x['User-Rating'], reverse=True)
    return render_template('MainR.html', genres=sorted(df['Genre'].unique()), songs=songs, selected_genre=user_genre, selected_sort=sort_order)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        if username:
            session['username'] = username
            return redirect(url_for('astar_page'))
        else:
            return render_template('login.html', error="Username cannot be empty.")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))

@app.route('/astar_redirect')
def astar_redirect():
    if 'username' in session:
        return redirect(url_for('astar_page'))
    else:
        return render_template('login_suggestion.html')

@app.route('/astar_page', methods=['GET', 'POST'])
def astar_page():
    if 'username' not in session:
        return redirect(url_for('login'))

    genres = sorted(df['Genre'].dropna().unique()) if 'Genre' in df.columns else []

    if request.method == 'POST':
        start_genre = request.form.get('start_genre', None)
        target_genre = request.form.get('target_genre', None)

        if start_genre and target_genre:
            recommendations = a_star_recommendation(start_genre, target_genre)
            return render_template(
                'astar.html',
                recommendations=recommendations,
                genres=genres,
                start_genre=start_genre,
                target_genre=target_genre,
            )
        else:
            flash("Please select both a start genre and a target genre!", "warning")

    return render_template('astar.html', genres=genres, start_genre=None, target_genre=None)

if __name__ == "__main__":
    app.run(debug=True)
