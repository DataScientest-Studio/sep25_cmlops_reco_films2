"""
API FastAPI pour le systeme de recommandation de films
Version PostgreSQL (Supabase)
"""
from fastapi import FastAPI, HTTPException, Depends, status
import mlflow
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
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from datetime import datetime, timedelta

load_dotenv()

# -----------------------------
# DATABASE
# -----------------------------

def get_connection():
    """
    Retourne une connexion PostgreSQL
    Force IPv4 pour éviter les problèmes de réseau dans Docker
    """
    db_host = os.getenv("DB_HOST")
    print(f"[INFO] Résolution DNS: {db_host}")

    try:
        ipv4_address = socket.getaddrinfo(db_host, None, socket.AF_INET)[0][4][0]
        print(f"[INFO] Résolution DNS: {db_host} -> {ipv4_address}")
    except Exception as e:
        print(f"[WARNING] Erreur résolution DNS: {e}, utilisation du hostname")
        ipv4_address = db_host
    
    return psycopg2.connect(
        host=ipv4_address,
        port=os.getenv("DB_PORT"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )

DATABASE_URL = os.getenv("DATABASE_URL")

# -----------------------------
# AUTHENTICATION JWT + BCRYPT
# -----------------------------

SECRET_KEY = os.getenv("SECRET_KEY", "ta_cle_secrete_ici")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Configuration JWT
SECRET_KEY = "ta_cle_secrete_ici"  # Remplace par une clé sécurisée et stocke-la dans .env
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Modèles Pydantic
class Token(BaseModel):
    access_token: str
    token_type: str
    userid: int

class TokenData(BaseModel):
    username: Optional[str] = None

class User(BaseModel):
    username: str
    disabled: Optional[bool] = None

class UserInDB(User):
    hashed_password: str

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str):
    return pwd_context.hash(password)

def get_user(db, username: str):
    if username in db:
        return UserInDB(**db[username])

def authenticate_user(username: str, password: str):
    user = get_user(fake_users_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta if expires_delta else timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=username)
    if user is None:
        raise credentials_exception
    return user

def get_random_userid():
    """Récupère un userid aléatoire depuis la base de données."""
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT userid FROM ratings ORDER BY RANDOM() LIMIT 1;")
            result = cur.fetchone()
            return result[0] if result else None
    finally:
        conn.close()

# -----------------------------
# FAKE USERS DB
# -----------------------------
fake_users_db = {
    "admin": {
        "username": "admin",
        # Mot de passe: RecoFilm!2025
        "hashed_password": "$2b$12$iDecWAqW5S59lSuKhNfSWuDenkh3/6SnoiJmJvJtIebWfiLGgju86",
        "disabled": False,
    }
}

# -----------------------------
# APP INITIALIZATION
# -----------------------------
app = FastAPI(
    title="RecoFilm API",
    description="API de recommandation de films basee sur MovieLens 20M",
    version="2.0.0"
)

Instrumentator().instrument(app).expose(app)

active_requests = Gauge(
    'http_requests_in_progress',
    'Number of HTTP requests currently being processed',
    registry=REGISTRY
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
TRAIN_SCRIPT = BASE_DIR / "train_model.py"
USER_MATRIX_PATH = BASE_DIR / "user_matrix.csv"
MOVIE_MATRIX_PATH = BASE_DIR / "movie_matrix.csv"

# -----------------------------
# Pydantic Models
# -----------------------------
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

class Token(BaseModel):
    access_token: str
    token_type: str
    userid: int

# -----------------------------
# ENDPOINTS
# -----------------------------

@app.get("/")
def read_root():
    return {
        "message": "Bienvenue sur l'API RecoFilm",
        "version": "2.0.0 (PostgreSQL)",
    }

@app.get("/health", response_model=HealthResponse)
def health_check():
    model_exists = (MODEL_DIR / "model.pkl").exists()
    try:
        conn = get_connection()
        conn.close()
        db_ok = True
    except:
        db_ok = False
    return {
        "status": "healthy" if (model_exists and db_ok) else "unhealthy",
        "model_loaded": model_exists,
        "database_connected": db_ok
    }

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer", "userid": get_random_userid()}

@app.post("/training", response_model=TrainingResponse)
def train_model():
    try:
        if not TRAIN_SCRIPT.exists():
            raise HTTPException(status_code=404, detail=f"Script d'entrainement non trouve: {TRAIN_SCRIPT}")
        result = subprocess.run([sys.executable, str(TRAIN_SCRIPT)], cwd=str(BASE_DIR))
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Erreur lors de l'entrainement")
        model_path = MODEL_DIR / "model.pkl"
        return {"status": "success", "message": "Modele entraine avec succes", "model_path": str(model_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.middleware("http")
async def track_active_requests(request, call_next):
    active_requests.inc()
    try:
        response = await call_next(request)
        return response
    finally:
        active_requests.dec()

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        user_id = request.userid
        num_recommendations = request.numRecommendations
        client = mlflow.MlflowClient()
        model_name = "recofilm-knn-recommender"
        champion = client.get_model_version_by_alias(model_name, "champion")
        run_id = champion.run_id

        local_dir = MODEL_DIR
        local_dir.mkdir(exist_ok=True)
        mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="", dst_path=str(local_dir))

        with open(local_dir / "model.pkl", "rb") as f:
            model = pickle.load(f)
        with open(local_dir / "movie_ids.pkl", "rb") as f:
            movie_ids = pickle.load(f)

        if not USER_MATRIX_PATH.exists():
            raise HTTPException(status_code=404, detail="user_matrix.csv non trouve.")
        user_matrix = pd.read_csv(USER_MATRIX_PATH)
        user_data = user_matrix[user_matrix['userid'] == user_id]
        if user_data.empty:
            raise HTTPException(status_code=404, detail=f"Utilisateur {user_id} non trouve")
        user_profile = user_data.drop('userid', axis=1).values[0]

        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT movieid FROM ratings WHERE userid = %s", (user_id,))
        watched_results = cursor.fetchall()
        watched_movies = set(row[0] for row in watched_results)

        distances, indices = model.kneighbors([user_profile])
        recommended_movie_ids = movie_ids[indices[0]]
        filtered_recommendations = [mid for mid in recommended_movie_ids if mid not in watched_movies][:num_recommendations]

        if not MOVIE_MATRIX_PATH.exists():
            raise HTTPException(status_code=404, detail="movie_matrix.csv non trouve.")
        movie_matrix = pd.read_csv(MOVIE_MATRIX_PATH)
        recommendations = []
        for movie_id in filtered_recommendations:
            cursor.execute("SELECT title, genres FROM movies WHERE movieid = %s", (int(movie_id),))
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

        return {"userid": user_id, "numRecommendations": len(recommendations), "recommendations": recommendations}

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)