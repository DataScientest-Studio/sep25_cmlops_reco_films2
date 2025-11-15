#!/usr/bin/env python3
"""
preprocess_movielens_sqlite_from_db.py

Préprocessing MovieLens 20M directement depuis la base SQLite.
Crée de nouvelles tables pour le training Surprise / modèles hybrides :
- ratings_preprocessed
- movies_preprocessed
- item_features
"""

import sqlite3
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

# -----------------------------
# CONFIG
# -----------------------------
DB_PATH = "../../database/movie_database.db"
MIN_USER_RATINGS = 5
MIN_MOVIE_RATINGS = 10

# -----------------------------
# ETAPE 1 : Charger les tables brutes depuis SQLite
# -----------------------------
def load_raw_data(conn):
    print("Lecture des tables brutes 'ratings' et 'movies' depuis SQLite...")
    ratings = pd.read_sql("SELECT * FROM ratings", conn)
    movies = pd.read_sql("SELECT * FROM movies", conn)
    print(f"Chargé {len(ratings):,} ratings et {len(movies):,} films")
    return ratings, movies

# -----------------------------
# ETAPE 2 : Filtrage
# -----------------------------
def filter_data(ratings):
    print("Filtrage des utilisateurs et films peu actifs...")
    user_counts = ratings['userId'].value_counts()
    movie_counts = ratings['movieId'].value_counts()
    valid_users = user_counts[user_counts >= MIN_USER_RATINGS].index
    valid_movies = movie_counts[movie_counts >= MIN_MOVIE_RATINGS].index
    filtered = ratings[
        ratings['userId'].isin(valid_users) &
        ratings['movieId'].isin(valid_movies)
    ].copy()
    print(f"Après filtrage : {len(filtered):,} notes, "
          f"{filtered['userId'].nunique():,} utilisateurs, "
          f"{filtered['movieId'].nunique():,} films.")
    return filtered

# -----------------------------
# ETAPE 3 : Préparer features items
# -----------------------------
def prepare_item_features(movies):
    print("Extraction des genres pour chaque film...")
    movies['genres'] = movies['genres'].replace('(no genres listed)', '').fillna('')
    movies['genres_list'] = movies['genres'].apply(lambda x: x.split('|') if x else [])

    # Stocker la liste des genres en string pour SQLite
    movies['genres_str'] = movies['genres_list'].apply(lambda x: '|'.join(x))

    mlb = MultiLabelBinarizer()
    genre_features = mlb.fit_transform(movies['genres_list'])
    genre_labels = mlb.classes_

    features_df = pd.DataFrame(genre_features, columns=genre_labels)
    features_df['movieId'] = movies['movieId'].values
    features_df = features_df[['movieId'] + list(genre_labels)]

    print(f"{len(genre_labels)} features créées pour les films")
    return features_df, movies

# -----------------------------
# ETAPE 4 : Sauvegarde dans SQLite
# -----------------------------
def save_preprocessed(conn, ratings, movies, item_features):
    print("Création des tables preprocessées dans SQLite...")

    # Supprimer la colonne list avant sauvegarde
    if 'genres_list' in movies.columns:
        movies = movies.drop(columns=['genres_list'])

    # Supprimer les tables existantes si elles existent
    tables_to_replace = ['ratings_preprocessed', 'movies_preprocessed', 'item_features']
    for table in tables_to_replace:
        conn.execute(f"DROP TABLE IF EXISTS {table}")

    # Écriture des tables
    ratings.to_sql("ratings_preprocessed", conn, index=False)
    movies.to_sql("movies_preprocessed", conn, index=False)
    item_features.to_sql("item_features", conn, index=False)

    print("✅ Tables 'ratings_preprocessed', 'movies_preprocessed', 'item_features' créées ou remplacées.")

# -----------------------------
# MAIN
# -----------------------------
def preprocess_from_db():
    conn = sqlite3.connect(DB_PATH)
    try:
        ratings, movies = load_raw_data(conn)
        ratings_filtered = filter_data(ratings)
        movies_filtered = movies[movies['movieId'].isin(ratings_filtered['movieId'].unique())]
        item_features, movies_filtered = prepare_item_features(movies_filtered)
        save_preprocessed(conn, ratings_filtered, movies_filtered, item_features)
    finally:
        conn.close()
        print("Connexion SQLite fermée.")

if __name__ == "__main__":
    preprocess_from_db()
