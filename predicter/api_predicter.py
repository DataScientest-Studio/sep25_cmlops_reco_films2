"""
app.py
API FastAPI pour prédire avec un modèle SVD exporté dans /models
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import pickle
import pandas as pd
import logging

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_FILE = MODELS_DIR / "svd_model.pkl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("api.log", mode="a")
    ]
)
logger = logging.getLogger(__name__)

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

def load_model():
    """Charge le modèle SVD exporté en local."""
    try:
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Modèle introuvable dans /models")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de chargement du modèle: {str(e)}")

# -----------------------------
# API ENDPOINTS
# -----------------------------
@app.post("/predict")
def predict(req: PredictRequest):
    """Prédit la note pour un user_id et movie_id donnés."""
    try:
        logger.info("Appel à l'endpoint /predict")
        model = load_model()
        logger.info("Modèle chargé depuis /models")

        input_df = pd.DataFrame({
            "user_id": [str(req.user_id)],
            "movie_id": [str(req.movie_id)]
        })

        prediction = model.predict(input_df)
        result = {
            "user_id": req.user_id,
            "movie_id": req.movie_id,
            "predicted_rating": round(float(prediction.iloc[0]), 2)
        }
        logger.info(f"Résultat final: {result}")
        return result

    except Exception as e:
        logger.exception("Erreur lors de la prédiction")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {str(e)}")