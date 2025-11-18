"""
app.py
API FastAPI pour entraîner et prédire un modèle de recommandation SVD avec PostgreSQL
"""
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from pathlib import Path
import pickle
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
from datetime import date
import psycopg2

# Charger les variables d'environnement
load_dotenv()

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_FILE = MODELS_DIR / "svd_model.pkl"
RATING_SCALE = (0.5, 5.0)
CHUNK_SIZE = 30

# Configuration PostgreSQL
DB_CONFIG = {
    "host": os.getenv("PGHOST", "crossover.proxy.rlwy.net"),
    "database": os.getenv("PGDATABASE", "railway"),
    "user": os.getenv("PGUSER", "postgres"),
    "password": os.getenv("PGPASSWORD", "GsoaNFyHnDBTGuebcvqIzEbuZTmSrtio"),
    "port": os.getenv("PGPORT", "25783"),
}

# Créer le dossier models si nécessaire
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# FASTAPI INITIALIZATION
# -----------------------------
app = FastAPI(title="Recommandation Film API")

# -----------------------------
# MODELS
# -----------------------------
class PredictRequest(BaseModel):
    user_id: int
    movie_id: int

class DataInsertRequest(BaseModel):
    force_insert: bool = False

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def get_db_engine():
    """Crée et retourne un moteur SQLAlchemy pour PostgreSQL."""
    db_url = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    return create_engine(db_url)

def get_db_connection():
    """Crée et retourne une connexion psycopg2 pour PostgreSQL."""
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False
    return conn

def load_ratings_from_db():
    """Charge les ratings depuis la base PostgreSQL."""
    engine = get_db_engine()
    try:
        ratings = pd.read_sql("SELECT userid, movieid, rating FROM ratings_preprocessed", engine)
        return ratings
    finally:
        engine.dispose()

def prepare_surprise_dataset(ratings):
    """Prépare le dataset pour Surprise."""
    reader = Reader(rating_scale=RATING_SCALE)
    data = Dataset.load_from_df(ratings[['userid', 'movieid', 'rating']], reader)
    return data

def train_svd_model():
    """Entraîne le modèle SVD et le sauvegarde."""
    ratings = load_ratings_from_db()
    data = prepare_surprise_dataset(ratings)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    algo = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02)
    algo.fit(trainset)
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)
    # Sauvegarde du modèle
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(algo, f)
    return {"rmse": rmse, "mae": mae, "model_path": str(MODEL_FILE.resolve())}

def load_model():
    """Charge le modèle sauvegardé."""
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Le modèle n'existe pas : {MODEL_FILE.resolve()}")
    with open(MODEL_FILE, "rb") as f:
        algo = pickle.load(f)
    return algo

def check_and_update_daily_counts(conn, force_insert=False):
    """Vérifie et met à jour la table daily_counts. Retourne True si une insertion est nécessaire."""
    today = date.today()
    with conn.cursor() as cur:
        # Vérifier si une ligne avec id = 1 existe
        cur.execute("SELECT id, date, count FROM daily_counts WHERE id = 1;")
        result = cur.fetchone()

        if result is None:
            # Insérer une nouvelle ligne avec id = 1
            insert_sql = "INSERT INTO daily_counts (id, date, count) VALUES (1, %s, 0) RETURNING count;"
            cur.execute(insert_sql, (today,))
            count = 0
            needs_insertion = True
        else:
            id, existing_date, count = result
            if force_insert or existing_date < today:
                # Mettre à jour la date et incrémenter le compteur
                update_sql = "UPDATE daily_counts SET date = %s, count = count + 1 WHERE id = 1 RETURNING count;"
                cur.execute(update_sql, (today,))
                count = cur.fetchone()[0]
                needs_insertion = True
            else:
                needs_insertion = False  # Même date, ne pas insérer de données
        conn.commit()
        return needs_insertion, count

def get_csv_file_size(table_name):
    """Retourne le nombre total de lignes dans le fichier CSV."""
    data_dir = BASE_DIR / "data" / "raw_data"
    csv_path = data_dir / f"{table_name}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier non trouvé: {csv_path}")

    # Compter le nombre de lignes dans le fichier CSV
    with open(csv_path, 'r') as f:
        num_lines = sum(1 for line in f) - 1  # Soustraire 1 pour l'en-tête
    return num_lines

def insert_data_chunk(conn, table_name, count):
    """Insère un chunk de données dans la table spécifiée."""
    start_idx = count * CHUNK_SIZE
    end_idx = start_idx + CHUNK_SIZE

    # Chemin vers les fichiers CSV
    data_dir = BASE_DIR / "data" / "raw_data"
    csv_path = data_dir / f"{table_name}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier non trouvé: {csv_path}")

    # Vérifier la taille totale du fichier CSV
    total_lines = get_csv_file_size(table_name)
    if start_idx >= total_lines:
        return 0  # Aucune ligne à insérer

    # Lire le chunk de données
    chunk = pd.read_csv(csv_path, skiprows=range(1, start_idx + 1), nrows=CHUNK_SIZE)

    if chunk.empty:
        return 0

    # Définir les colonnes attendues pour chaque table
    table_columns = {
        "ratings": ["userId", "movieId", "rating", "timestamp"],
        "tags": ["userId", "movieId", "tag", "timestamp"],
        "genome-scores": ["movieId", "tagId", "relevance"]
    }

    if table_name not in table_columns:
        raise ValueError(f"Table inconnue: {table_name}")

    expected_columns = table_columns[table_name]

    # Vérifier que les colonnes attendues existent dans le chunk
    missing_columns = [col for col in expected_columns if col not in chunk.columns]
    if missing_columns:
        raise ValueError(f"Colonnes manquantes dans le fichier CSV: {missing_columns}")

    # Insérer les données dans la table
    with conn.cursor() as cur:
        if table_name == "ratings":
            for _, row in chunk.iterrows():
                cur.execute(
                    "INSERT INTO ratings (userid, movieid, rating, timestamp) VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING;",
                    (row['userId'], row['movieId'], row['rating'], row['timestamp'])
                )
        elif table_name == "tags":
            for _, row in chunk.iterrows():
                cur.execute(
                    "INSERT INTO tags (userid, movieid, tag, timestamp) VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING;",
                    (row['userId'], row['movieId'], row['tag'], row['timestamp'])
                )
        elif table_name == "genome-scores":
            for _, row in chunk.iterrows():
                cur.execute(
                    "INSERT INTO genome_scores (movieid, tagid, relevance) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING;",
                    (row['movieId'], row['tagId'], row['relevance'])
                )

    conn.commit()
    return len(chunk)

# -----------------------------
# API ENDPOINTS
# -----------------------------
@app.post("/training")
def training():
    """
    Entraîne le modèle SVD depuis la base PostgreSQL et le sauvegarde.
    """
    try:
        result = train_svd_model()
        return {"status": "success", "metrics": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'entraînement: {str(e)}")

@app.post("/predict")
def predict(req: PredictRequest):
    """
    Prédit la note pour un user_id et movie_id donnés.
    """
    try:
        algo = load_model()
        prediction = algo.predict(req.user_id, req.movie_id)
        return {
            "user_id": req.user_id,
            "movie_id": req.movie_id,
            "predicted_rating": round(prediction.est, 2)
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {str(e)}")

@app.post("/insert-data")
def insert_data(request: DataInsertRequest = Body(...)):
    """
    Insère un chunk de données dans les tables ratings, tags et genome_scores.
    Vérifie et met à jour daily_counts avant l'insertion.
    """
    conn = None
    try:
        conn = get_db_connection()
        needs_insertion, count = check_and_update_daily_counts(conn, request.force_insert)

        if not needs_insertion and not request.force_insert:
            return {
                "status": "no_insertion_needed",
                "message": "La date est la même que celle du jour, aucune insertion effectuée.",
                "count": count
            }

        tables = ["ratings", "tags", "genome-scores"]
        results = {}

        for table in tables:
            try:
                inserted_rows = insert_data_chunk(conn, table, count)
                results[table] = {
                    "inserted_rows": inserted_rows,
                    "start_idx": count * CHUNK_SIZE,
                    "end_idx": count * CHUNK_SIZE + CHUNK_SIZE
                }
            except Exception as e:
                results[table] = {
                    "error": str(e)
                }

        return {
            "status": "success",
            "count": count,
            "results": results
        }
    except Exception as e:
        if conn:
            conn.rollback()
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'insertion: {str(e)}")
    finally:
        if conn:
            conn.close()

@app.get("/daily-counts")
def get_daily_counts():
    """
    Récupère les informations de daily_counts.
    """
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT id, date, count FROM daily_counts WHERE id = 1;")
            result = cur.fetchone()
            if result is None:
                return {"id": 1, "date": None, "count": 0}
            id, date_val, count = result
            return {"id": id, "date": date_val, "count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération des daily_counts: {str(e)}")
    finally:
        if conn:
            conn.close()
