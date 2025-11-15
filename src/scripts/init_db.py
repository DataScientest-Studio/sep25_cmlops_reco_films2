#!/usr/bin/env python3
"""
create_movielens_sqlite.py
Crée movie_database.db et importe les fichiers CSV du dataset MovieLens 20M.

Prérequis:
    pip install pandas tqdm
Usage:
    python create_movielens_sqlite.py
"""

import sqlite3
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np

DB_PATH = Path("../../database/movie_database.db")
CSV_PATH = Path("../../data/raw_data/")
CSV_FILES = {
    "movies": Path(CSV_PATH, "movies.csv"),
    "ratings": Path(CSV_PATH, "ratings.csv"),
    "tags": Path(CSV_PATH, "tags.csv"),
    "genome_scores": Path(CSV_PATH, "genome-scores.csv"),
    "genome_tags": Path(CSV_PATH, "genome-tags.csv"),
    "links": Path(CSV_PATH, "links.csv"),  # optionnel si présent
}

CHUNK_SIZE = 200_000  # ajuste si besoin (mémoire / vitesse)

CREATE_TABLES_SQL = """
PRAGMA foreign_keys = ON;

-- users are implicit via userId in ratings/tags (no separate users file in ML-20M)
CREATE TABLE IF NOT EXISTS movies (
    movieId INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    genres TEXT
);

CREATE TABLE IF NOT EXISTS ratings (
    userId INTEGER NOT NULL,
    movieId INTEGER NOT NULL,
    rating REAL NOT NULL,
    timestamp INTEGER,
    FOREIGN KEY(movieId) REFERENCES movies(movieId)
);

CREATE TABLE IF NOT EXISTS tags (
    userId INTEGER NOT NULL,
    movieId INTEGER NOT NULL,
    tag TEXT NOT NULL,
    timestamp INTEGER,
    FOREIGN KEY(movieId) REFERENCES movies(movieId)
);

CREATE TABLE IF NOT EXISTS genome_scores (
    movieId INTEGER NOT NULL,
    tagId INTEGER NOT NULL,
    relevance REAL NOT NULL,
    FOREIGN KEY(movieId) REFERENCES movies(movieId)
);

CREATE TABLE IF NOT EXISTS genome_tags (
    tagId INTEGER PRIMARY KEY,
    tag TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS links (
    movieId INTEGER PRIMARY KEY,
    imdbId TEXT,
    tmdbId TEXT,
    FOREIGN KEY(movieId) REFERENCES movies(movieId)
);
"""

INDEX_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_ratings_user ON ratings(userId);",
    "CREATE INDEX IF NOT EXISTS idx_ratings_movie ON ratings(movieId);",
    "CREATE INDEX IF NOT EXISTS idx_tags_user ON tags(userId);",
    "CREATE INDEX IF NOT EXISTS idx_tags_movie ON tags(movieId);",
    "CREATE INDEX IF NOT EXISTS idx_genome_movie ON genome_scores(movieId);",
    "CREATE INDEX IF NOT EXISTS idx_genome_tag ON genome_scores(tagId);",
]

def connect_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA temp_store = MEMORY;")
    return conn

def create_schema(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.executescript(CREATE_TABLES_SQL)
    conn.commit()

# Colonnes "obligatoires" (NOT NULL) par table - utilisées pour le nettoyage.
REQUIRED_COLS = {
    "movies": ["movieId", "title"],
    "ratings": ["userId", "movieId", "rating"],
    "tags": ["userId", "movieId", "tag"],
    "genome_scores": ["movieId", "tagId", "relevance"],
    "genome_tags": ["tagId", "tag"],
    "links": ["movieId"],
}

def _to_native_py(value):
    """Convertit les types pandas/numpy en types natifs Python pour sqlite3."""
    if pd.isna(value):
        return None
    # numpy integer/float -> cast en int/float natif
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    # pandas Timestamp -> int (timestamp) ou str
    if isinstance(value, pd.Timestamp):
        return int(value.timestamp())
    return value

def import_csv_to_table(conn: sqlite3.Connection, csv_path: Path, table: str, dtype_map=None, use_cols=None):
    """
    Importe un CSV en chunks vers la table SQLite, avec préprocessing pour:
     - strip des chaînes,
     - remplacer chaînes vides par NA,
     - conversions numériques avec coercion,
     - suppression des lignes violant les colonnes NOT NULL connues.
    dtype_map: dict (ex: {"userId": int, "movieId": int})
    use_cols: colonnes à lire (optionnel).
    """
    if not csv_path.exists():
        print(f"[SKIP] {csv_path} absent.")
        return

    reader = pd.read_csv(csv_path, chunksize=CHUNK_SIZE, dtype=dtype_map, usecols=use_cols, low_memory=False)
    total_rows = 0
    total_inserted = 0
    total_skipped = 0
    cur = conn.cursor()
    insert_sql = None

    # Determine insert SQL based on table name (assure l'ordre des colonnes)
    if table == "movies":
        insert_sql = "INSERT OR IGNORE INTO movies(movieId, title, genres) VALUES (?, ?, ?)"
        cols = ["movieId", "title", "genres"]
    elif table == "ratings":
        insert_sql = "INSERT INTO ratings(userId, movieId, rating, timestamp) VALUES (?, ?, ?, ?)"
        cols = ["userId", "movieId", "rating", "timestamp"]
    elif table == "tags":
        insert_sql = "INSERT INTO tags(userId, movieId, tag, timestamp) VALUES (?, ?, ?, ?)"
        cols = ["userId", "movieId", "tag", "timestamp"]
    elif table == "genome_scores":
        insert_sql = "INSERT INTO genome_scores(movieId, tagId, relevance) VALUES (?, ?, ?)"
        cols = ["movieId", "tagId", "relevance"]
    elif table == "genome_tags":
        insert_sql = "INSERT OR IGNORE INTO genome_tags(tagId, tag) VALUES (?, ?)"
        cols = ["tagId", "tag"]
    elif table == "links":
        insert_sql = "INSERT OR IGNORE INTO links(movieId, imdbId, tmdbId) VALUES (?, ?, ?)"
        cols = ["movieId", "imdbId", "tmdbId"]
    else:
        raise ValueError("Table inconnue: " + table)

    required = REQUIRED_COLS.get(table, [])

    print(f"Import {csv_path} -> {table} ...")
    for chunk in tqdm(reader, desc=f"chunks {table}"):
        # Normalize column names
        chunk.columns = [c.strip() for c in chunk.columns]

        # Keep only expected columns (avoid colonnes inattendues)
        chunk = chunk[[c for c in cols if c in chunk.columns]]

        # If a column is missing entirely in this CSV, add it with NA
        for c in cols:
            if c not in chunk.columns:
                chunk[c] = pd.NA

        # Replace pure-empty strings (spaces, '') by NA
        chunk = chunk.replace(r'^\s*$', pd.NA, regex=True)

        # Strip whitespace for object/string columns
        for c in chunk.select_dtypes(include="object").columns:
            # use .astype(str) carefully: preserve NA
            chunk[c] = chunk[c].where(chunk[c].notna(), pd.NA)
            # strip on non-null values
            chunk.loc[chunk[c].notna(), c] = chunk.loc[chunk[c].notna(), c].str.strip()

        # Apply dtype coercion for numeric types defined in dtype_map
        if dtype_map:
            for col, typ in dtype_map.items():
                if col not in chunk.columns:
                    continue
                if typ in (int,):
                    # coerce to numeric then to pandas nullable Int64
                    chunk[col] = pd.to_numeric(chunk[col], errors="coerce").astype("Int64")
                elif typ in (float,):
                    chunk[col] = pd.to_numeric(chunk[col], errors="coerce").astype("Float64")
                elif typ in (str,):
                    # ensure strings, keep NA as NA
                    chunk[col] = chunk[col].where(chunk[col].notna(), pd.NA)
                    # ensure type object
                    chunk[col] = chunk[col].astype(object)

        # Drop rows missing required NOT NULL columns
        before = len(chunk)
        if required:
            chunk = chunk.dropna(subset=[c for c in required if c in chunk.columns])
        after_drop = len(chunk)
        skipped_in_chunk = before - after_drop

        # Prepare rows: convert pandas NA / numpy types -> native Python types (None, int, float, str)
        rows = []
        for row in chunk[cols].itertuples(index=False, name=None):
            py_row = tuple(_to_native_py(v) for v in row)
            rows.append(py_row)

        # Insert and commit; handle/skip problematic rows gracefully
        inserted = 0
        if rows:
            try:
                cur.executemany(insert_sql, rows)
                conn.commit()
                inserted = len(rows)
            except sqlite3.IntegrityError as e:
                # Si IntegrityError (p.ex. violation clé), essaye insertion ligne par ligne en skipant les fautives
                conn.rollback()
                inserted = 0
                for r in rows:
                    try:
                        cur.execute(insert_sql, r)
                        inserted += 1
                    except sqlite3.IntegrityError:
                        # skip problematic row
                        continue
                conn.commit()

        total_rows += before
        total_inserted += inserted
        total_skipped += skipped_in_chunk + (before - inserted if inserted <= before else 0)

    print(f"Import fini: {total_inserted} lignes insérées dans {table}.")
    if total_skipped:
        print(f"  ({total_skipped} lignes sautées pendant le préprocessing ou à cause d'erreurs)")

def create_indexes(conn: sqlite3.Connection):
    cur = conn.cursor()
    for sql in INDEX_SQL:
        cur.execute(sql)
    conn.commit()
    print("Index créés.")

def main():
    conn = connect_db(DB_PATH)
    print("Connected to", DB_PATH)
    create_schema(conn)

    # Import small files first
    import_csv_to_table(conn, CSV_FILES["movies"], "movies",
                        dtype_map={"movieId": int, "title": str, "genres": str})
    import_csv_to_table(conn, CSV_FILES["genome_tags"], "genome_tags",
                        dtype_map={"tagId": int, "tag": str})
    import_csv_to_table(conn, CSV_FILES["links"], "links",
                        dtype_map={"movieId": int, "imdbId": str, "tmdbId": str})

    # Large files: chunked
    import_csv_to_table(conn, CSV_FILES["genome_scores"], "genome_scores",
                        dtype_map={"movieId": int, "tagId": int, "relevance": float},
                        use_cols=["movieId", "tagId", "relevance"])
    import_csv_to_table(conn, CSV_FILES["ratings"], "ratings",
                        dtype_map={"userId": int, "movieId": int, "rating": float, "timestamp": int},
                        use_cols=["userId", "movieId", "rating", "timestamp"])
    import_csv_to_table(conn, CSV_FILES["tags"], "tags",
                        dtype_map={"userId": int, "movieId": int, "tag": str, "timestamp": int},
                        use_cols=["userId", "movieId", "tag", "timestamp"])

    create_indexes(conn)
    conn.close()
    print("Base SQLite créée avec succès:", DB_PATH)

if __name__ == "__main__":
    main()
