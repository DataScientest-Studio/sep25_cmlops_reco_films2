"""
train_surprise_from_sqlite.py

Entraîne un modèle de recommandation Surprise directement
depuis les tables preprocessées dans SQLite.
"""

import sqlite3
import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split
from pathlib import Path
import pickle
import sys

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # remonte 3 niveaux
DB_PATH = BASE_DIR / "database" / "movie_database.db"
MODELS_DIR = BASE_DIR / "models"
MODEL_FILE = MODELS_DIR / "svd_model.pkl"
RATING_SCALE = (0.5, 5.0)  # MovieLens rating scale

# Créer le dossier models s'il n'existe pas
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Vérification que la DB existe
if not DB_PATH.exists():
    print(f"❌ Le fichier DB n'existe pas : {DB_PATH.resolve()}")
    sys.exit(1)
else:
    print(f"✅ DB trouvée : {DB_PATH.resolve()}")

# -----------------------------
# ETAPE 1 : Charger les données depuis SQLite
# -----------------------------
def load_ratings_from_db(db_path):
    print("📥 Chargement des données depuis SQLite...")
    conn = sqlite3.connect(db_path)
    ratings = pd.read_sql("SELECT userId, movieId, rating FROM ratings_preprocessed", conn)
    conn.close()
    print(f"✅ Chargé {len(ratings):,} ratings depuis 'ratings_preprocessed'")
    return ratings

# -----------------------------
# ETAPE 2 : Préparer dataset Surprise
# -----------------------------
def prepare_surprise_dataset(ratings):
    print("📊 Préparation du dataset pour Surprise...")

    # Logs sur le DataFrame
    print(f"- Nombre de ratings : {len(ratings):,}")
    print(f"- Colonnes : {ratings.columns.tolist()}")
    print(f"- Aperçu des 5 premières lignes :\n{ratings.head()}")

    # Vérification des colonnes nécessaires
    required_cols = ['userId', 'movieId', 'rating']
    for col in required_cols:
        if col not in ratings.columns:
            raise ValueError(f"La colonne '{col}' est manquante dans le DataFrame.")

    print(f"- Utilisation du rating scale : {RATING_SCALE}")
    reader = Reader(rating_scale=RATING_SCALE)
    data = Dataset.load_from_df(ratings[required_cols], reader)
    print("✅ Dataset préparé avec succès pour Surprise.")
    return data

# -----------------------------
# ETAPE 3 : Entraînement du modèle
# -----------------------------
def train_model(data):
    print("⚡ Split train/test 80/20...")
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    print("🏋️ Entraînement du modèle SVD...")
    algo = SVD(n_factors=50, n_epochs=1, lr_all=0.005, reg_all=0.02)
    algo.fit(trainset)
    print("🏋️ Fin de l'entrainement !")
   # print("🔍 Évaluation sur le test set...")
    #predictions = algo.test(testset)
    #rmse = accuracy.rmse(predictions)
    #mae = accuracy.mae(predictions)
    #print(f"📈 RMSE : {rmse:.4f}, MAE : {mae:.4f}")

    return algo

# -----------------------------
# MAIN
# -----------------------------
def main():
    print("🚀 Début de l'entraînement du modèle Surprise depuis SQLite")
    ratings = load_ratings_from_db(DB_PATH)
    data = prepare_surprise_dataset(ratings)
    algo = train_model(data)
    print("✅ Entraînement terminé.")

    # Sauvegarde du modèle
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(algo, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"💾 Modèle sauvegardé dans {MODEL_FILE.resolve()}")

if __name__ == "__main__":
    main()