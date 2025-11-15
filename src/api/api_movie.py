"""
app.py

API FastAPI pour entraîner et prédire un modèle de recommandation SVD
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import pickle
import sqlite3
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DB_PATH = BASE_DIR / "database" / "movie_database.db"
MODELS_DIR = BASE_DIR / "models"
MODEL_FILE = MODELS_DIR / "svd_model.pkl"
RATING_SCALE = (0.5, 5.0)

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

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def load_ratings_from_db(db_path):
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path.resolve()}")
    conn = sqlite3.connect(db_path)
    ratings = pd.read_sql("SELECT userId, movieId, rating FROM ratings_preprocessed", conn)
    conn.close()
    return ratings

def prepare_surprise_dataset(ratings):
    reader = Reader(rating_scale=RATING_SCALE)
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    return data

def train_svd_model():
    ratings = load_ratings_from_db(DB_PATH)
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
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Le modèle n'existe pas : {MODEL_FILE.resolve()}")
    with open(MODEL_FILE, "rb") as f:
        algo = pickle.load(f)
    return algo

# -----------------------------
# API ENDPOINTS
# -----------------------------
@app.post("/training")
def training():
    """
    Entraîne le modèle SVD depuis la base SQLite et le sauvegarde.
    """
    try:
        result = train_svd_model()
        return {"status": "success", "metrics": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
        raise HTTPException(status_code=500, detail=str(e))
