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
import mlflow
import mlflow.sklearn
import logging
from mlflow import MlflowClient

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_FILE = MODELS_DIR / "svd_model.pkl"
RATING_SCALE = (0.5, 5.0)
CHUNK_SIZE = 30


# Charger les variables d'environnement
load_dotenv(dotenv_path=BASE_DIR)

DB_CONFIG = {
    "host": os.getenv("PGHOST"),
    "database": os.getenv("PGDATABASE"),
    "user": os.getenv("PGUSER"),
    "password": os.getenv("PGPASSWORD"),
    "port": os.getenv("PGPORT"),
}

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("reco_movie_api")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),  # affiche dans la console
        logging.FileHandler("training.log", mode="a")  # sauvegarde dans un fichier
    ]
)

logger = logging.getLogger(__name__)

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

class SurpriseSVDWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, algo):
        self.algo = algo

    def predict(self, context, model_input: pd.DataFrame):
        # Vérifier que les colonnes attendues existent
        if not {"user_id", "movie_id"}.issubset(model_input.columns):
            raise ValueError("Le DataFrame doit contenir les colonnes 'user_id' et 'movie_id'")

        preds = []
        for _, row in model_input.iterrows():
            est = self.algo.predict(str(row["user_id"]), str(row["movie_id"])).est
            preds.append(est)

        # Retourner une Series pour compatibilité avec .iloc
        return pd.Series(preds)




# ------------------- Méthodes utilitaires -------------------

def load_and_prepare_data():
    """Charge les ratings depuis la base et prépare le dataset Surprise."""
    ratings = load_ratings_from_db()
    data = prepare_surprise_dataset(ratings)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    return ratings, trainset, testset


def train_and_evaluate(trainset, testset, params):
    """Entraîne le modèle SVD et calcule les métriques."""
    algo = SVD(**params)
    algo.fit(trainset)
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)
    return algo, rmse, mae


def log_model_and_metrics(algo, params, rmse, mae, run):
    """Log les paramètres, métriques et le modèle dans MLflow."""
    mlflow.log_params(params)
    mlflow.log_metrics({"rmse": rmse, "mae": mae})

    # Sauvegarde locale
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(algo, f)
    mlflow.log_artifact(str(MODEL_FILE.resolve()), artifact_path="models")

    input_example = pd.DataFrame({"user_id": ["u1"], "movie_id": ["i1"]})

    # Log du modèle dans le run
    artifact_path = "svd_model_artifact"
    mlflow.pyfunc.log_model(
        artifact_path=artifact_path,
        python_model=SurpriseSVDWrapper(algo),
        input_example=input_example,
        signature=mlflow.models.infer_signature(input_example, [3.5])
    )
    return artifact_path


def promote_model_with_comparison(client, run, artifact_path, rmse, mae):
    """
    Compare le nouveau modèle avec celui en production.
    Si le nouveau est meilleur sur RMSE ET MAE, il passe en Production.
    Sinon, il reste en Staging.
    """
    try:
        client.create_registered_model("svd_model")
    except Exception:
        pass  # déjà créé

    # Créer une nouvelle version dans le Model Registry
    model_uri = f"runs:/{run.info.run_id}/{artifact_path}"
    mv = client.create_model_version(
        name="svd_model",
        source=model_uri,
        run_id=run.info.run_id
    )
    new_version = int(mv.version)

    # Récupérer le modèle actuellement en production
    try:
        prod_version_info = client.get_model_version_by_alias("svd_model", "production")
    except Exception:
        prod_version_info = None

    if prod_version_info:
        prod_version = int(prod_version_info.version)
        prod_run_id = prod_version_info.run_id

        # Charger les métriques du modèle en production
        prod_run = client.get_run(prod_run_id)
        prod_rmse = float(prod_run.data.metrics.get("rmse", 9999))
        prod_mae = float(prod_run.data.metrics.get("mae", 9999))

        # Comparaison multi-métriques
        if rmse < prod_rmse and mae < prod_mae:
            # Nouveau meilleur sur les deux métriques → promotion
            client.set_registered_model_alias("svd_model", "production", new_version)
            client.set_registered_model_alias("svd_model", "staging", prod_version)
            stage = "Production"
            logger.info(f"Nouveau modèle (v{new_version}) promu en Production. Ancien (v{prod_version}) rétrogradé en Staging.")
        else:
            # Nouveau moins bon → staging
            client.set_registered_model_alias("svd_model", "staging", new_version)
            stage = "Staging"
            logger.info(f"Nouveau modèle (v{new_version}) placé en Staging. Production reste v{prod_version}.")
    else:
        # Aucun modèle en production → premier modèle devient Production
        client.set_registered_model_alias("svd_model", "production", new_version)
        stage = "Production"
        logger.info(f"Premier modèle (v{new_version}) promu en Production.")

    return stage





# ------------------- Fonction principale -------------------

def train_svd_model():
    """Pipeline complet d'entraînement, log et promotion du modèle SVD."""
    logger.info("===== Début de l'entraînement du modèle SVD =====")

    try:
        # 1. Charger et préparer les données
        ratings, trainset, testset = load_and_prepare_data()

        # 2. Définir les hyperparamètres
        params = {"n_factors": 50, "n_epochs": 20, "lr_all": 0.005, "reg_all": 0.02}

        # 3. Entraîner et évaluer
        with mlflow.start_run() as run:
            algo, rmse, mae = train_and_evaluate(trainset, testset, params)

            # 4. Logger modèle et métriques
            artifact_path = log_model_and_metrics(algo, params, rmse, mae, run)

            # 5. Promotion automatique
            client = MlflowClient()
            stage = promote_model_with_comparison(client, run, artifact_path, rmse, mae)

        logger.info(f"Modèle promu automatiquement avec alias {stage}")
        logger.info("===== Fin de l'entraînement du modèle SVD =====")

        return {
            "rmse": rmse,
            "mae": mae,
            "model_path": str(MODEL_FILE.resolve()),
            "mlflow_run_id": run.info.run_id,
            "params": params,
            "n_ratings": len(ratings),
            "alias": stage
        }

    except Exception as e:
        logger.error(f"Erreur pendant l'entraînement : {e}")
        raise




def load_model():
    """
    Charge le modèle SVD en Production depuis le MLflow Model Registry.
    Retourne l'objet PythonModel prêt à être utilisé pour la prédiction.
    """
    try:
        # Charger le modèle en Production
        model = mlflow.pyfunc.load_model("models:/svd_model@production")
        return model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Impossible de charger le modèle Production: {str(e)}")



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
        logger.info("Appel à l'endpoint /predict")
        logger.debug(f"Requête reçue: user_id={req.user_id}, movie_id={req.movie_id}")

        model = load_model()
        logger.info("Modèle chargé depuis MLflow")

        input_df = pd.DataFrame({
            "user_id": [str(req.user_id)],
            "movie_id": [str(req.movie_id)]
        })
        logger.debug(f"DataFrame construit pour la prédiction: {input_df}")

        prediction = model.predict(input_df)
        logger.debug(f"Résultat brut du modèle: {prediction}")

        result = {
            "user_id": req.user_id,
            "movie_id": req.movie_id,
            "predicted_rating": round(float(prediction.iloc[0]), 2)
        }
        logger.info(f"Résultat final: {result}")

        return result

    except FileNotFoundError as e:
        logger.error(f"Modèle introuvable: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Erreur lors de la prédiction")
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
