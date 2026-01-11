"""
app.py
API FastAPI pour prédire avec un modèle SVD chargé
directement depuis MLflow Model Registry (alias Production)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import logging
import os
import mlflow
import mlflow.pyfunc
from functools import lru_cache

# -----------------------------
# CONFIGURATION
# -----------------------------

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI","http://mlflow:5000")

if not MLFLOW_TRACKING_URI:
    raise RuntimeError("La variable d'environnement MLFLOW_TRACKING_URI n'est pas définie")

MODEL_URI = "models:/svd_model@production"

# -----------------------------
# LOGGING
# -----------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# -----------------------------
# FASTAPI INITIALIZATION
# -----------------------------

app = FastAPI(
    title="Recommandation Film API",
    description="API de prédiction utilisant le modèle MLflow en Production",
    version="1.0.0"
)

# -----------------------------
# SCHEMAS
# -----------------------------

class PredictRequest(BaseModel):
    userid: int
    movieid: int

class PredictResponse(BaseModel):
    userid: int
    movieid: int
    predicted_rating: float

# -----------------------------
# MODEL LOADING
# -----------------------------

@lru_cache(maxsize=1)
def load_model():
    """
    Charge le modèle depuis MLflow Model Registry (alias Production).
    Le modèle est mis en cache mémoire.
    """
    try:
        logger.info("Connexion à MLflow...")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        logger.info(f"Chargement du modèle MLflow : {MODEL_URI}")
        model = mlflow.pyfunc.load_model(MODEL_URI)

        logger.info("Modèle chargé avec succès depuis MLflow")
        return model

    except Exception as e:
        logger.exception("Erreur lors du chargement du modèle MLflow")
        raise RuntimeError(str(e))

# -----------------------------
# API ENDPOINTS
# -----------------------------

@app.get("/")
def root():
    return {"status": "ok", "message": "Recommandation Film API is running"}

@app.get("/health")
def health():
    """
    Vérifie que l'API et le modèle MLflow sont opérationnels.
    """
    try:
        load_model()
        return {"status": "healthy", "model": "loaded"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Prédit la note pour un utilisateur et un film donnés.
    """
    try:
        model = load_model()

        input_df = pd.DataFrame({
            "userid": [req.userid],
            "movieid": [req.movieid]
        })

        prediction = model.predict(input_df)

        return {
            "userid": req.userid,
            "movieid": req.movieid,
            "predicted_rating": round(float(prediction.iloc[0]), 2)
        }

    except Exception as e:
        logger.exception("Erreur lors de la prédiction")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reload-model")
def reload_model():
    """
    Force le rechargement du modèle depuis MLflow (utile après une promotion).
    """
    try:
        load_model.cache_clear()
        load_model()
        return {"status": "success", "message": "Modèle rechargé depuis MLflow"}
    except Exception as e:
        logger.exception("Erreur lors du rechargement du modèle")
        raise HTTPException(status_code=500, detail=str(e))
