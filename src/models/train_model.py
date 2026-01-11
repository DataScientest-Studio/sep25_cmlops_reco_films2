"""
train_surprise_from_sqlite_mlflow.py

Entraîne un modèle de recommandation Surprise depuis SQLite
et loggue les résultats dans MLflow (tracking + registry).
"""

import sqlite3
import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split
from pathlib import Path
import pickle
import sys
import mlflow
import mlflow.pyfunc

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DB_PATH = BASE_DIR / "database" / "movie_database.db"
MODELS_DIR = BASE_DIR / "models"
MODEL_FILE = MODELS_DIR / "svd_model.pkl"
RATING_SCALE = (0.5, 5.0)

MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Indiquer que tu veux utiliser le serveur MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("reco_film")


if not DB_PATH.exists():
    print(f"❌ Le fichier DB n'existe pas : {DB_PATH.resolve()}")
    sys.exit(1)
else:
    print(f"✅ DB trouvée : {DB_PATH.resolve()}")

# -----------------------------
# ETAPE 1 : Charger les données
# -----------------------------
def load_ratings_from_db(db_path):
    conn = sqlite3.connect(db_path)
    ratings = pd.read_sql("SELECT userId, movieId, rating FROM ratings_preprocessed", conn)
    conn.close()
    return ratings

# -----------------------------
# ETAPE 2 : Préparer dataset Surprise
# -----------------------------
def prepare_surprise_dataset(ratings):
    reader = Reader(rating_scale=RATING_SCALE)
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    return data

# -----------------------------
# ETAPE 3 : Entraînement + MLflow
# -----------------------------
def train_model(data):
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    algo = SVD(n_factors=50, n_epochs=5, lr_all=0.005, reg_all=0.02)
    algo.fit(trainset)

    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)

    return algo, rmse, mae

# -----------------------------
# MAIN avec MLflow
# -----------------------------
def main():
    ratings = load_ratings_from_db(DB_PATH)
    data = prepare_surprise_dataset(ratings)

    with mlflow.start_run(run_name="SVD_surprise_sqlite"):
        algo, rmse, mae = train_model(data)

        # Log hyperparamètres
        mlflow.log_param("n_factors", 50)
        mlflow.log_param("n_epochs", 20)
        mlflow.log_param("lr_all", 0.005)
        mlflow.log_param("reg_all", 0.02)

        # Log métriques
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)

        # Sauvegarde locale
        with open(MODEL_FILE, "wb") as f:
            pickle.dump(algo, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Log modèle dans MLflow
        mlflow.log_artifact(MODEL_FILE, artifact_path="model")

        print(f"✅ Modèle entraîné et loggué dans MLflow. RMSE={rmse:.4f}, MAE={mae:.4f}")

if __name__ == "__main__":
    main()