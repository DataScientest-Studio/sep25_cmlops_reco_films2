#!/usr/bin/env python3
"""
init_db.py
Crée les tables MovieLens et une table daily_counts dans une base PostgreSQL (ex: Railway).
Prérequis:
    pip install pandas psycopg2-binary tqdm
Usage:
    python init_db.py
"""
import psycopg2
from psycopg2 import sql
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration PostgreSQL
DB_CONFIG = {
    "host": os.getenv("PGHOST", "crossover.proxy.rlwy.net"),
    "database": os.getenv("PGDATABASE", "railway"),
    "user": os.getenv("PGUSER", "postgres"),
    "password": os.getenv("PGPASSWORD", "GsoaNFyHnDBTGuebcvqIzEbuZTmSrtio"),
    "port": os.getenv("PGPORT", "25783"),
}

# Chemin absolu du répertoire du script
SCRIPT_DIR = Path(__file__).parent

# Chemin vers le dossier des données
DATA_DIR = SCRIPT_DIR / "data" / "raw_data"
if not DATA_DIR.exists():
    DATA_DIR = SCRIPT_DIR / "../../data/raw_data"

CSV_FILES = {
    "movies": DATA_DIR / "movies.csv",
    "ratings": DATA_DIR / "ratings.csv",
    "tags": DATA_DIR / "tags.csv",
    "genome_scores": DATA_DIR / "genome-scores.csv",
    "genome_tags": DATA_DIR / "genome-tags.csv",
    "links": DATA_DIR / "links.csv",
}

CHUNK_SIZE = 1000

# SQL pour créer les tables
CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS movies (
    movieId INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    genres TEXT
);
CREATE TABLE IF NOT EXISTS genome_tags (
    tagId INTEGER PRIMARY KEY,
    tag TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ratings (
    userId INTEGER NOT NULL,
    movieId INTEGER NOT NULL,
    rating REAL NOT NULL,
    timestamp BIGINT,
    FOREIGN KEY(movieId) REFERENCES movies(movieId) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS tags (
    userId INTEGER NOT NULL,
    movieId INTEGER NOT NULL,
    tag TEXT NOT NULL,
    timestamp BIGINT,
    FOREIGN KEY(movieId) REFERENCES movies(movieId) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS genome_scores (
    movieId INTEGER NOT NULL,
    tagId INTEGER NOT NULL,
    relevance REAL NOT NULL,
    FOREIGN KEY(movieId) REFERENCES movies(movieId) ON DELETE CASCADE,
    FOREIGN KEY(tagId) REFERENCES genome_tags(tagId) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS links (
    movieId INTEGER PRIMARY KEY,
    imdbId TEXT,
    tmdbId TEXT,
    FOREIGN KEY(movieId) REFERENCES movies(movieId) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS daily_counts (
    id SERIAL PRIMARY KEY,
    date DATE UNIQUE NOT NULL,
    count INTEGER NOT NULL
);
"""

# Index pour PostgreSQL
INDEX_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_ratings_user ON ratings(userId);",
    "CREATE INDEX IF NOT EXISTS idx_ratings_movie ON ratings(movieId);",
    "CREATE INDEX IF NOT EXISTS idx_tags_user ON tags(userId);",
    "CREATE INDEX IF NOT EXISTS idx_tags_movie ON tags(movieId);",
    "CREATE INDEX IF NOT EXISTS idx_genome_movie ON genome_scores(movieId);",
    "CREATE INDEX IF NOT EXISTS idx_genome_tag ON genome_scores(tagId);",
]

# Colonnes obligatoires (NOT NULL) par table
REQUIRED_COLS = {
    "movies": ["movieId", "title"],
    "ratings": ["userId", "movieId", "rating"],
    "tags": ["userId", "movieId", "tag"],
    "genome_scores": ["movieId", "tagId", "relevance"],
    "genome_tags": ["tagId", "tag"],
    "links": ["movieId"],
    "daily_counts": ["date", "count"],
}

def connect_db():
    """Se connecte à PostgreSQL (Railway)."""
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False
    return conn

def create_schema(conn):
    """Crée les tables dans PostgreSQL."""
    with conn.cursor() as cur:
        cur.execute(CREATE_TABLES_SQL)
    conn.commit()
    print("Tables créées.")

def _to_native_py(value):
    """Convertit les types pandas/numpy en types natifs Python pour psycopg2."""
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return int(value.timestamp())
    return value

def import_csv_to_table(conn, csv_path, table, dtype_map=None, use_cols=None):
    """Importe un CSV vers PostgreSQL en chunks."""
    if not csv_path.exists():
        print(f"[SKIP] {csv_path} absent.")
        return
    insert_sql = None
    cols = None
    if table == "movies":
        insert_sql = "INSERT INTO movies (movieId, title, genres) VALUES (%s, %s, %s) ON CONFLICT (movieId) DO NOTHING;"
        cols = ["movieId", "title", "genres"]
    elif table == "ratings":
        insert_sql = "INSERT INTO ratings (userId, movieId, rating, timestamp) VALUES (%s, %s, %s, %s);"
        cols = ["userId", "movieId", "rating", "timestamp"]
    elif table == "tags":
        insert_sql = "INSERT INTO tags (userId, movieId, tag, timestamp) VALUES (%s, %s, %s, %s);"
        cols = ["userId", "movieId", "tag", "timestamp"]
    elif table == "genome_scores":
        insert_sql = "INSERT INTO genome_scores (movieId, tagId, relevance) VALUES (%s, %s, %s);"
        cols = ["movieId", "tagId", "relevance"]
    elif table == "genome_tags":
        insert_sql = "INSERT INTO genome_tags (tagId, tag) VALUES (%s, %s) ON CONFLICT (tagId) DO NOTHING;"
        cols = ["tagId", "tag"]
    elif table == "links":
        insert_sql = "INSERT INTO links (movieId, imdbId, tmdbId) VALUES (%s, %s, %s) ON CONFLICT (movieId) DO NOTHING;"
        cols = ["movieId", "imdbId", "tmdbId"]
    else:
        raise ValueError(f"Table inconnue: {table}")

    required = REQUIRED_COLS.get(table, [])
    print(f"Import {csv_path} -> {table} ...")

    # Limiter à 30000 lignes pour les tables ratings, tags et genome_scores
    nrows = 30000 if table in ["ratings", "tags", "genome_scores"] else None

    reader = pd.read_csv(csv_path, chunksize=CHUNK_SIZE, dtype=dtype_map, usecols=use_cols, low_memory=False, nrows=nrows)
    total_rows = 0
    total_inserted = 0
    total_skipped = 0
    with conn.cursor() as cur:
        for chunk in tqdm(reader, desc=f"chunks {table}"):
            chunk.columns = [c.strip() for c in chunk.columns]
            chunk = chunk[[c for c in cols if c in chunk.columns]]
            for c in chunk.select_dtypes(include="object").columns:
                chunk[c] = chunk[c].where(chunk[c].notna(), pd.NA).astype(object)
            if dtype_map:
                for col, typ in dtype_map.items():
                    if col not in chunk.columns:
                        continue
                    if typ in (int,):
                        chunk[col] = pd.to_numeric(chunk[col], errors="coerce").astype("Int64")
                    elif typ in (float,):
                        chunk[col] = pd.to_numeric(chunk[col], errors="coerce").astype("Float64")
                    elif typ in (str,):
                        chunk[col] = chunk[col].where(chunk[col].notna(), pd.NA).astype(object)
            before = len(chunk)
            if required:
                chunk = chunk.dropna(subset=[c for c in required if c in chunk.columns])
            after_drop = len(chunk)
            skipped_in_chunk = before - after_drop
            rows = [_to_native_py(v) for row in chunk[cols].itertuples(index=False, name=None) for v in row]
            rows = [tuple(rows[i:i + len(cols)]) for i in range(0, len(rows), len(cols))]
            inserted = 0
            if rows:
                try:
                    cur.executemany(insert_sql, rows)
                    conn.commit()
                    inserted = len(rows)
                except psycopg2.IntegrityError as e:
                    conn.rollback()
                    inserted = 0
                    for r in rows:
                        try:
                            cur.execute(insert_sql, r)
                            inserted += 1
                        except psycopg2.IntegrityError:
                            continue
                    conn.commit()
            total_rows += before
            total_inserted += inserted
            total_skipped += skipped_in_chunk + (before - inserted if inserted <= before else 0)
    print(f"Import fini: {total_inserted} lignes insérées dans {table}.")
    if total_skipped:
        print(f"  ({total_skipped} lignes sautées)")


def init_daily_counts(conn):
    """Initialise la table daily_counts avec la date du jour et un compteur à 0."""
    with conn.cursor() as cur:
        insert_sql = "INSERT INTO daily_counts (date, count) VALUES (%s, %s) ON CONFLICT (date) DO NOTHING;"
        today = pd.Timestamp.now().date()
        try:
            cur.execute(insert_sql, (today, 0))
            conn.commit()
            print(f"Ligne initialisée dans daily_counts: {today}, count=0")
        except Exception as e:
            conn.rollback()
            print(f"Erreur lors de l'initialisation de daily_counts: {e}")

def create_indexes(conn):
    """Crée les index dans PostgreSQL."""
    with conn.cursor() as cur:
        for sql in INDEX_SQL:
            cur.execute(sql)
    conn.commit()
    print("Index créés.")

def main():
    conn = connect_db()
    print(f"Connecté à PostgreSQL: {DB_CONFIG['host']}/{DB_CONFIG['database']}")
    create_schema(conn)
    # Initialiser la table daily_counts
    init_daily_counts(conn)
    import_csv_to_table(conn, CSV_FILES["movies"], "movies", dtype_map={"movieId": int, "title": str, "genres": str})
    import_csv_to_table(conn, CSV_FILES["genome_tags"], "genome_tags", dtype_map={"tagId": int, "tag": str})
    import_csv_to_table(conn, CSV_FILES["links"], "links", dtype_map={"movieId": int, "imdbId": str, "tmdbId": str})
    import_csv_to_table(conn, CSV_FILES["genome_scores"], "genome_scores", dtype_map={"movieId": int, "tagId": int, "relevance": float})
    import_csv_to_table(conn, CSV_FILES["ratings"], "ratings", dtype_map={"userId": int, "movieId": int, "rating": float, "timestamp": int})
    import_csv_to_table(conn, CSV_FILES["tags"], "tags", dtype_map={"userId": int, "movieId": int, "tag": str, "timestamp": int})

    create_indexes(conn)
    conn.close()
    print("Base PostgreSQL créée avec succès.")

if __name__ == "__main__":
    main()
