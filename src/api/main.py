"""
API FastAPI pour le systeme de recommandation de films
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import pickle
import sqlite3
import subprocess
from pathlib import Path
import sys


app = FastAPI(
    title="RecoFilm API",
    description="API de recommandation de films basee sur MovieLens 20M",
    version="1.0.0"
)


# Chemins globaux
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
DB_PATH = PROJECT_ROOT / "database" / "recofilm.db"
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


# Root endpoint
@app.get("/")
def read_root():
    """
    Page d'accueil de l'API
    """
    return {
        "message": "Bienvenue sur l'API RecoFilm",
        "version": "1.0.0",
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
        conn = sqlite3.connect(DB_PATH)
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
        
        model_path = MODEL_DIR / "model.pkl"
        
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
        user_id = request.userId
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
        user_matrix = pd.read_csv(USER_MATRIX_PATH)
        user_data = user_matrix[user_matrix['userId'] == user_id]
        
        if user_data.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Utilisateur {user_id} non trouve"
            )
        
        # Extraire le profil utilisateur
        user_profile = user_data.drop('userId', axis=1).values[0]
        
        # Recuperer les films deja vus
        conn = sqlite3.connect(DB_PATH)
        query = f"SELECT DISTINCT movieId FROM ratings WHERE userId = {user_id}"
        watched = pd.read_sql_query(query, conn)
        watched_movies = set(watched['movieId'].values)
        
        # Trouver les films similaires
        distances, indices = model.kneighbors([user_profile])
        recommended_movie_ids = movie_ids[indices[0]]
        
        # Filtrer les films deja vus
        filtered_recommendations = [
            movie_id for movie_id in recommended_movie_ids 
            if movie_id not in watched_movies
        ][:num_recommendations]
        
        # Recuperer les infos des films
        movie_matrix = pd.read_csv(MOVIE_MATRIX_PATH)
        recommendations = []
        
        for movie_id in filtered_recommendations:
            query = f"SELECT title, genres FROM movies WHERE movieId = {movie_id}"
            movie_info = pd.read_sql_query(query, conn)
            
            if not movie_info.empty:
                movie_row = movie_matrix[movie_matrix['movieId'] == movie_id]
                
                recommendations.append({
                    "movieId": int(movie_id),
                    "title": movie_info['title'].values[0],
                    "genres": movie_info['genres'].values[0],
                    "avg_rating": float(movie_row['avg_rating'].values[0]) if not movie_row.empty else 0.0,
                    "num_ratings": int(movie_row['num_ratings'].values[0]) if not movie_row.empty else 0
                })
        
        conn.close()
        
        return {
            "userId": user_id,
            "numRecommendations": len(recommendations),
            "recommendations": recommendations
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    print("Demarrage de l'API RecoFilm...")
    print("Documentation disponible sur: http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000)