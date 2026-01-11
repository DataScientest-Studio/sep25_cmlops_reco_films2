#!/usr/bin/env python3
"""
preprocess_movielens_postgres_from_db.py
Préprocessing MovieLens 20M directement depuis la base PostgreSQL.
Crée de nouvelles tables pour le training Surprise / modèles hybrides :
- ratings_preprocessed
- movies_preprocessed
- item_features
"""
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
from io import StringIO
import psycopg2

# Charger les variables d'environnement
load_dotenv()

# -----------------------------
# CONFIG
# -----------------------------
DB_CONFIG = {
    "host": os.getenv("PGHOST", "crossover.proxy.rlwy.net"),
    "database": os.getenv("PGDATABASE", "railway"),
    "user": os.getenv("PGUSER", "postgres"),
    "password": os.getenv("PGPASSWORD", "GsoaNFyHnDBTGuebcvqIzEbuZTmSrtio"),
    "port": os.getenv("PGPORT", "25783"),
}

MIN_USER_RATINGS = 5
MIN_MOVIE_RATINGS = 10

# -----------------------------
# Fonction pour créer une connexion SQLAlchemy
# -----------------------------
def get_db_engine():
    """Crée et retourne un moteur SQLAlchemy pour PostgreSQL."""
    db_url = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    engine = create_engine(db_url)
    return engine

# -----------------------------
# Fonction pour se connecter à PostgreSQL avec psycopg2
# -----------------------------
def connect_db():
    """Se connecte à PostgreSQL avec psycopg2."""
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False
    return conn

# -----------------------------
# ETAPE 1 : Charger les tables brutes depuis PostgreSQL
# -----------------------------
def load_raw_data(engine):
    print("Lecture des tables brutes 'ratings' et 'movies' depuis PostgreSQL...")
    ratings = pd.read_sql("SELECT * FROM ratings", engine)
    movies = pd.read_sql("SELECT * FROM movies", engine)
    print(f"Chargé {len(ratings):,} ratings et {len(movies):,} films")
    return ratings, movies

# -----------------------------
# ETAPE 2 : Filtrage
# -----------------------------
def filter_data(ratings):
    print("Filtrage des utilisateurs et films peu actifs...")
    user_counts = ratings['userid'].value_counts()
    movie_counts = ratings['movieid'].value_counts()
    valid_users = user_counts[user_counts >= MIN_USER_RATINGS].index
    valid_movies = movie_counts[movie_counts >= MIN_MOVIE_RATINGS].index
    filtered = ratings[
        ratings['userid'].isin(valid_users) &
        ratings['movieid'].isin(valid_movies)
    ].copy()
    print(f"Après filtrage : {len(filtered):,} notes, "
          f"{filtered['userid'].nunique():,} utilisateurs, "
          f"{filtered['movieid'].nunique():,} films.")
    return filtered

# -----------------------------
# ETAPE 3 : Préparer features items
# -----------------------------
def prepare_item_features(movies):
    print("Extraction des genres pour chaque film...")
    movies = movies.copy()
    movies.loc[:, 'genres'] = movies['genres'].replace('(no genres listed)', '').fillna('')
    movies.loc[:, 'genres_list'] = movies['genres'].apply(lambda x: x.split('|') if x else [])
    mlb = MultiLabelBinarizer()
    genre_features = mlb.fit_transform(movies['genres_list'])
    genre_labels = mlb.classes_
    features_df = pd.DataFrame(genre_features, columns=genre_labels, dtype=bool)
    features_df['movieid'] = movies['movieid'].values
    features_df = features_df[['movieid'] + list(genre_labels)]
    print(f"{len(genre_labels)} features créées pour les films")
    return features_df, movies[['movieid', 'title', 'genres']]

# -----------------------------
# ETAPE 4 : Sauvegarde dans PostgreSQL
# -----------------------------
def save_preprocessed(conn, ratings, movies, item_features):
    print("Création des tables preprocessées dans PostgreSQL...")

    with conn.cursor() as cur:
        # Supprimer les tables existantes si elles existent
        tables_to_replace = ['ratings_preprocessed', 'movies_preprocessed', 'item_features']
        for table in tables_to_replace:
            cur.execute(f'DROP TABLE IF EXISTS "{table}";')

        # Créer la table ratings_preprocessed
        cur.execute("""
        CREATE TABLE ratings_preprocessed (
            userid INTEGER,
            movieid INTEGER,
            rating REAL,
            timestamp BIGINT
        );
        """)

        # Créer la table movies_preprocessed
        cur.execute("""
        CREATE TABLE movies_preprocessed (
            movieid INTEGER PRIMARY KEY,
            title TEXT,
            genres TEXT
        );
        """)

        # Créer dynamiquement la table item_features en fonction des colonnes disponibles
        columns = ['movieid INTEGER'] + [f'"{col}" BOOLEAN' if '-' in col else f'{col} BOOLEAN' for col in item_features.columns[1:]]
        columns_sql = ', '.join(columns)
        cur.execute(f"""
        CREATE TABLE item_features (
            {columns_sql},
            PRIMARY KEY (movieid)
        );
        """)

    # Écrire les données
    with conn.cursor() as cur:
        # ratings_preprocessed
        output = StringIO()
        ratings.to_csv(output, sep='\t', header=False, index=False)
        output.seek(0)
        cur.copy_expert(sql="COPY ratings_preprocessed FROM STDIN WITH (FORMAT CSV, DELIMITER '\t')", file=output)
        conn.commit()

        # movies_preprocessed
        output = StringIO()
        movies.to_csv(output, sep='\t', header=False, index=False)
        output.seek(0)
        cur.copy_expert(sql="COPY movies_preprocessed FROM STDIN WITH (FORMAT CSV, DELIMITER '\t')", file=output)
        conn.commit()

        # item_features
        output = StringIO()
        item_features.to_csv(output, sep='\t', header=False, index=False)
        output.seek(0)
        cur.copy_expert(sql="COPY item_features FROM STDIN WITH (FORMAT CSV, DELIMITER '\t')", file=output)
        conn.commit()

    print("✅ Tables 'ratings_preprocessed', 'movies_preprocessed', 'item_features' créées ou remplacées.")

# -----------------------------
# MAIN
# -----------------------------
def preprocess_from_db():
    engine = get_db_engine()
    conn = connect_db()
    try:
        ratings, movies = load_raw_data(engine)
        ratings_filtered = filter_data(ratings)
        movies_filtered = movies[movies['movieid'].isin(ratings_filtered['movieid'].unique())].copy()
        item_features, movies_with_features = prepare_item_features(movies_filtered)
        save_preprocessed(conn, ratings_filtered, movies_with_features, item_features)
    finally:
        conn.close()
        engine.dispose()
        print("Connexions PostgreSQL fermées.")

if __name__ == "__main__":
    preprocess_from_db()
