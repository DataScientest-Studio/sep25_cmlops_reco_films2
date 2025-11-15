"""
predict_surprise_model.py

Script pour prédire les notes avec un modèle SVD entraîné
en utilisant la bibliothèque Surprise.
"""

import pickle
from pathlib import Path
from surprise import SVD

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # ajuster si nécessaire
MODELS_DIR = BASE_DIR / "models"
MODEL_FILE = MODELS_DIR / "svd_model.pkl"

# Vérification que le modèle existe
if not MODEL_FILE.exists():
    raise FileNotFoundError(f"Le modèle n'existe pas : {MODEL_FILE.resolve()}")

# -----------------------------
# CHARGEMENT DU MODÈLE
# -----------------------------
print(f"📥 Chargement du modèle depuis {MODEL_FILE.resolve()}")
with open(MODEL_FILE, "rb") as f:
    algo: SVD = pickle.load(f)
print("✅ Modèle chargé avec succès.")

# -----------------------------
# FONCTION DE PRÉDICTION
# -----------------------------
def predict_rating(user_id: int, movie_id: int):
    """
    Prédit la note qu'un utilisateur donnerait à un film.
    
    Args:
        user_id (int): ID de l'utilisateur
        movie_id (int): ID du film

    Returns:
        float: note prédite
    """
    prediction = algo.predict(user_id, movie_id)
    print(f"Utilisateur {user_id} -> Film {movie_id} : Note prédite = {prediction.est:.2f}")
    return prediction.est

# -----------------------------
# EXEMPLES D'UTILISATION
# -----------------------------
if __name__ == "__main__":
    # Exemple simple
    predict_rating(1, 101)
    predict_rating(2, 103)
    
    # Exemple avec une liste d'utilisateurs et films
    test_cases = [
        (1, 105),
        (3, 101),
        (5, 110),
    ]
    for user_id, movie_id in test_cases:
        predict_rating(user_id, movie_id)
