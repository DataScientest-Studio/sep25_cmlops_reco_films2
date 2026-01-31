"""
API FastAPI pour le systeme de recommandation de films
Version PostgreSQL (Supabase)
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from prometheus_fastapi_instrumentator import Instrumentator
import pandas as pd
import pickle
import subprocess
from pathlib import Path
import sys
from prometheus_client import Gauge, REGISTRY
import os
import socket
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_batch

load_dotenv()

def get_connection():
    """
    Retourne une connexion PostgreSQL
    Force IPv4 pour éviter les problèmes de réseau dans Docker
    """
    db_host = os.getenv("DB_HOST")
    # Résoudre le hostname en IPv4 seulement
    print(f"[INFO] Résolution DNS: {db_host}")

    try:
        ipv4_address = socket.getaddrinfo(
            db_host, 
            None, 
            socket.AF_INET  # Force IPv4
        )[0][4][0]
        print(f"[INFO] Résolution DNS: {db_host} -> {ipv4_address}")
    except Exception as e:
        print(f"[WARNING] Erreur résolution DNS: {e}, utilisation du hostname")
        ipv4_address = db_host
    
    return psycopg2.connect(
        host=ipv4_address,  # Utilise l'adresse IPv4 résolue
        port=os.getenv("DB_PORT"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )


app = FastAPI(
    title="RecoFilm API",
    description="API de recommandation de films basee sur MovieLens 20M",
    version="2.0.0"
)

# Instrumenter Prometheus
Instrumentator().instrument(app).expose(app)

# Gauge pour les requêtes actives
active_requests = Gauge(
    'http_requests_in_progress',
    'Number of HTTP requests currently being processed',
    registry=REGISTRY
)

# Chemins globaux
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
TRAIN_SCRIPT = BASE_DIR / "train_model.py"
USER_MATRIX_PATH = BASE_DIR / "user_matrix.csv"
MOVIE_MATRIX_PATH = BASE_DIR / "movie_matrix.csv"


# Modeles Pydantic pour validation
class PredictionRequest(BaseModel):
    userid: int
    numRecommendations: Optional[int] = 10


class MovieRecommendation(BaseModel):
    movieid: int
    title: str
    genres: str
    avg_rating: float
    num_ratings: int


class PredictionResponse(BaseModel):
    userid: int
    numRecommendations: int
    recommendations: List[MovieRecommendation]


class TrainingResponse(BaseModel):
    status: str
    message: str
    model_path: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    database_connected: bool


# Root endpoint
@app.get("/")
def read_root():
    """
    Page d'accueil de l'API
    """
    return {
        "message": "Bienvenue sur l'API RecoFilm",
        "version": "2.0.0 (PostgreSQL)",
        "endpoints": {
            "/docs": "Documentation interactive Swagger",
            "/health": "Verifier l'etat de l'API",
            "/training": "POST - Entrainer le modele",
            "/predict": "POST - Obtenir des recommandations"
        }
    }


# Health check
@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Verifie l'etat de l'API et des ressources
    """
    model_exists = (MODEL_DIR / "model.pkl").exists()
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        conn.close()
        db_ok = True
    except:
        db_ok = False
    
    return {
        "status": "healthy" if (model_exists and db_ok) else "unhealthy",
        "model_loaded": model_exists,
        "database_connected": db_ok
    }


# Training endpoint
@app.post("/training", response_model=TrainingResponse)
def train_model():
    """
    Lance l'entrainement du modele KNN
    """
    try:
        print("Lancement de l'entrainement du modele...")
        
        # Chemin vers le script d'entrainement
        train_script = TRAIN_SCRIPT

        if not train_script.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Script d'entrainement non trouve: {train_script}"
            )
        print("Execution  du modele...")
        # Executer le script d'entrainement
        result = subprocess.run(
            [sys.executable, str(train_script)],
            cwd=str(BASE_DIR)
        )
        print("Fin execution du modele...")
        
        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Erreur lors de l'entrainement: {result.stderr}"
            )
        
        model_path = MODEL_DIR / "model.pkl"
        print("model path : {model_path}")
        return {
            "status": "success",
            "message": "Modele entraine avec succes",
            "model_path": str(model_path)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.middleware("http")
async def track_active_requests(request, call_next):
    """
    Middleware pour tracker le nombre de requêtes actives
    """
    active_requests.inc()  # Incrémenter au début de la requête
    try:
        response = await call_next(request)
        return response
    finally:
        active_requests.dec()

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Genere des recommandations de films pour un utilisateur
    """
    try:
        user_id = request.userid
        num_recommendations = request.numRecommendations
        
        print(f"Generation de {num_recommendations} recommandations pour l'utilisateur {user_id}...")
        
        # Verifier que le modele existe
        model_path = MODEL_DIR / "model.pkl"
        ids_path = MODEL_DIR / "movie_ids.pkl"
        
        if not model_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Modele non trouve. Executez /training d'abord."
            )
        
        # Charger le modele
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(ids_path, 'rb') as f:
            movie_ids = pickle.load(f)
        
        # Charger user_matrix
        if not USER_MATRIX_PATH.exists():
            raise HTTPException(
                status_code=404,
                detail="user_matrix.csv non trouve. Executez preprocess.py d'abord."
            )
        
        user_matrix = pd.read_csv(USER_MATRIX_PATH)
        user_data = user_matrix[user_matrix['userid'] == user_id]
        
        if user_data.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Utilisateur {user_id} non trouve"
            )
        
        # Extraire le profil utilisateur
        user_profile = user_data.drop('userid', axis=1).values[0]
        
        # Connexion PostgreSQL
        conn = get_connection()
        cursor = conn.cursor()
        
        # Recuperer les films deja vus (requête paramétrée)
        cursor.execute(
            "SELECT DISTINCT movieid FROM ratings WHERE userid = %s",
            (user_id,)
        )
        watched_results = cursor.fetchall()
        watched_movies = set(row[0] for row in watched_results)
        
        # Trouver les films similaires
        distances, indices = model.kneighbors([user_profile])
        recommended_movie_ids = movie_ids[indices[0]]
        
        # Filtrer les films deja vus
        filtered_recommendations = [
            movie_id for movie_id in recommended_movie_ids 
            if movie_id not in watched_movies
        ][:num_recommendations]
        
        # Recuperer les infos des films
        if not MOVIE_MATRIX_PATH.exists():
            raise HTTPException(
                status_code=404,
                detail="movie_matrix.csv non trouve."
            )
        
        movie_matrix = pd.read_csv(MOVIE_MATRIX_PATH)
        recommendations = []
        
        for movie_id in filtered_recommendations:
            # Requête paramétrée pour récupérer les infos du film
            cursor.execute(
                "SELECT title, genres FROM movies WHERE movieid = %s",
                (int(movie_id),)
            )
            movie_result = cursor.fetchone()
            
            if movie_result:
                title, genres = movie_result
                movie_row = movie_matrix[movie_matrix['movieid'] == movie_id]
                
                recommendations.append({
                    "movieid": int(movie_id),
                    "title": title,
                    "genres": genres,
                    "avg_rating": float(movie_row['avg_rating'].values[0]) if not movie_row.empty else 0.0,
                    "num_ratings": int(movie_row['num_ratings'].values[0]) if not movie_row.empty else 0
                })
        
        cursor.close()
        conn.close()
        
        return {
            "userid": user_id,
            "numRecommendations": len(recommendations),
            "recommendations": recommendations
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Erreur dans /predict: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    print("Demarrage de l'API RecoFilm...")
    print("Documentation disponible sur: http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000)