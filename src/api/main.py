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
import time

# Imports pour reload
import threading
from datetime import datetime
import joblib


# Ajout pour PostgreSQL
sys.path.append(str(Path(__file__).parent.parent.parent / "database"))
from config import get_connection


# Variables globales pour le rechargement du modèle
last_reload_time = None
model_lock = threading.Lock()

# Variables globales pour le modèle
model = None
movie_ids = None
movies_df = None


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
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
USER_MATRIX_PATH = PROJECT_ROOT / "data" / "processed" / "user_matrix.csv"
MOVIE_MATRIX_PATH = PROJECT_ROOT / "data" / "processed" / "movie_matrix.csv"


# Modeles Pydantic pour validation
class PredictionRequest(BaseModel):
    userId: int
    numRecommendations: Optional[int] = 10


class MovieRecommendation(BaseModel):
    movieId: int
    title: str
    genres: str
    avg_rating: float
    num_ratings: int


class PredictionResponse(BaseModel):
    userId: int
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
    last_model_reload: Optional[str] = None


def load_model_from_disk():
    """
    Charge le modèle et les données associées depuis le disque
    
    Returns:
        tuple: (model, movie_ids, movies_df)
    """
    global last_reload_time
    
    print("🔄 Chargement du modèle...")
    
    model_path = MODEL_DIR / "model.pkl"
    movie_ids_path = MODEL_DIR / "movie_ids.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Modèle non trouvé: {model_path}")
    
    if not movie_ids_path.exists():
        raise FileNotFoundError(f"Movie IDs non trouvés: {movie_ids_path}")
    
    # Charger le modèle
    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)
    
    with open(movie_ids_path, 'rb') as f:
        loaded_movie_ids = pickle.load(f)
    
    # Charger les métadonnées des films depuis la DB
    conn = get_connection()
    loaded_movies_df = pd.read_sql_query("SELECT movieId, title, genres FROM movies", conn)
    loaded_movies_df.columns = ['movieId', 'title', 'genres']
    conn.close()
    
    last_reload_time = datetime.now()
    print(f"✅ Modèle chargé avec succès à {last_reload_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    return loaded_model, loaded_movie_ids, loaded_movies_df


# Chargement initial au démarrage
print("🚀 Démarrage de l'API...")
try:
    model, movie_ids, movies_df = load_model_from_disk()
except Exception as e:
    print(f"⚠️ Impossible de charger le modèle au démarrage: {e}")
    model, movie_ids, movies_df = None, None, None


# ============================================================================
# MIDDLEWARE
# ============================================================================
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


# ============================================================================
# ENDPOINTS
# ============================================================================

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
            "/predict": "POST - Obtenir des recommandations",
            "/reload": "POST - Recharger le modele sans redemarrer"
        }
    }


# Health check (AMÉLIORÉ avec info reload)
@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Verifie l'etat de l'API et des ressources
    """
    model_loaded = model is not None
    
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
        "status": "healthy" if (model_loaded and db_ok) else "degraded",
        "model_loaded": model_loaded,
        "database_connected": db_ok,
        "last_model_reload": last_reload_time.isoformat() if last_reload_time else None
    }


# NOUVEL ENDPOINT: Reload model
@app.post("/reload", tags=["Admin"])
async def reload_model():
    """
    Recharge le modèle sans redémarrer l'API
    
    Utile après un réentraînement pour mettre à jour le modèle en production
    sans interruption de service (zero-downtime deployment).
    
    Returns:
        dict: Statut du rechargement avec timestamps et informations du modèle
    
    Raises:
        HTTPException: Si le modèle ne peut pas être chargé
    """
    global model, movie_ids, movies_df
    
    try:
        old_reload_time = last_reload_time
        
        # Recharger avec un lock pour éviter les conflits
        with model_lock:
            model, movie_ids, movies_df = load_model_from_disk()
        
        return {
            "status": "success",
            "message": "Modèle rechargé avec succès",
            "previous_load": old_reload_time.isoformat() if old_reload_time else None,
            "current_load": last_reload_time.isoformat(),
            "model_info": {
                "n_samples": model.n_samples_fit_ if hasattr(model, 'n_samples_fit_') else "N/A",
                "n_features": model.n_features_in_ if hasattr(model, 'n_features_in_') else "N/A",
                "n_movies": len(movie_ids) if movie_ids is not None else 0
            }
        }
    
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Modèle non trouvé: {str(e)}"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du rechargement: {str(e)}"
        )


# Training endpoint
@app.post("/training", response_model=TrainingResponse)
def train_model():
    """
    Lance l'entrainement du modele KNN
    """
    try:
        print("Lancement de l'entrainement du modele...")
        
        # Chemin vers le script d'entrainement
        train_script = PROJECT_ROOT / "src" / "models" / "train_model.py"
        
        if not train_script.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Script d'entrainement non trouve: {train_script}"
            )
        
        # Executer le script d'entrainement
        result = subprocess.run(
            [sys.executable, str(train_script)],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT)
        )
        
        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Erreur lors de l'entrainement: {result.stderr}"
            )
        
        model_path = MODEL_DIR / "knn_recommender.pkl"
        
        return {
            "status": "success",
            "message": "Modele entraine avec succes",
            "model_path": str(model_path)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Genere des recommandations de films pour un utilisateur
    """
    try:
        # Vérifier que le modèle est chargé
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Modèle non chargé. Utilisez /reload ou redémarrez l'API."
            )
        
        user_id = request.userId
        num_recommendations = request.numRecommendations
        
        print(f"Generation de {num_recommendations} recommandations pour l'utilisateur {user_id}...")
        
        # Charger user_matrix
        if not USER_MATRIX_PATH.exists():
            raise HTTPException(
                status_code=404,
                detail="user_matrix.csv non trouve. Executez preprocess.py d'abord."
            )
        
        user_matrix = pd.read_csv(USER_MATRIX_PATH)
        user_data = user_matrix[user_matrix['userId'] == user_id]
        
        if user_data.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Utilisateur {user_id} non trouve"
            )
        
        # Extraire le profil utilisateur
        user_profile = user_data.drop('userId', axis=1).values[0]
        
        # Connexion PostgreSQL
        conn = get_connection()
        cursor = conn.cursor()
        
        # Recuperer les films deja vus (requête paramétrée)
        cursor.execute(
            "SELECT DISTINCT movieId FROM ratings WHERE userId = %s",
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
                "SELECT title, genres FROM movies WHERE movieId = %s",
                (int(movie_id),)
            )
            movie_result = cursor.fetchone()
            
            if movie_result:
                title, genres = movie_result
                movie_row = movie_matrix[movie_matrix['movieId'] == movie_id]
                
                recommendations.append({
                    "movieId": int(movie_id),
                    "title": title,
                    "genres": genres,
                    "avg_rating": float(movie_row['avg_rating'].values[0]) if not movie_row.empty else 0.0,
                    "num_ratings": int(movie_row['num_ratings'].values[0]) if not movie_row.empty else 0
                })
        
        cursor.close()
        conn.close()
        
        return {
            "userId": user_id,
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