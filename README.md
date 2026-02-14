# RecoFilm — Système de Recommandation de Films

Système de recommandation de films type « Netflix » développé dans le cadre de la formation DataScientest (module MLOps). L'application combine **filtrage collaboratif** (KNN, SVD) et **content-based** pour générer des recommandations personnalisées à partir d'un identifiant utilisateur.

---

## Table des matières

- [Vue d'ensemble](#vue-densemble)
- [Architecture](#architecture)
- [Ports et services](#ports-et-services)
- [Prérequis](#prérequis)
- [Installation et démarrage](#installation-et-démarrage)
- [Fonctionnement](#fonctionnement)
- [Variables d'environnement](#variables-denvironnement)
- [Structure du projet](#structure-du-projet)

---

## Vue d'ensemble

RecoFilm permet à un utilisateur de saisir son identifiant (`user_id`) et d'obtenir une liste de films recommandés (Top-N). Le système s'appuie sur :

- **Données** : MovieLens 20M (ratings, movies, tags)
- **Base de données** : PostgreSQL (Supabase)
- **Modèles** : KNN (collaborative filtering) et SVD (matrix factorization)
- **MLOps** : MLflow (suivi des expériences), Airflow (orchestration), Docker (conteneurisation)

---

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Streamlit UI  │────▶│   KNN API        │────▶│   PostgreSQL    │
│   (8501)        │     │   (8002)         │     │   (Supabase)    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │                         │
         │                         │
         ▼                         ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Predicter API  │     │   Trainer API    │     │   MLflow        │
│   (8001)         │     │   (8000)         │     │   (5000)        │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │                         │
         └────────────┬────────────┘
                      │
                      ▼
              ┌──────────────────┐
              │   Airflow        │
              │   (8085)         │
              └──────────────────┘
```

---

## Ports et services

| Service | Port hôte | Port conteneur | Description |
|---------|-----------|----------------|-------------|
| **MLflow** | 5000 | 5000 | Suivi des expériences ML, registry des modèles |
| **Trainer API** | 8000 | 8000 | Entraînement SVD, insertion des données |
| **Predicter API** | 8001 | 8000 | Prédictions SVD (recommandations) |
| **KNN API** | 8002 | 8000 | Entraînement KNN, prédictions, authentification JWT |
| **Streamlit UI** | 8501 | 8501 | Interface utilisateur de démonstration |
| **Airflow** | 8085 | 8080 | Orchestration des pipelines (training, insert-data) |
| **PostgreSQL** | — | 5432 | Base de données (interne au réseau Docker) |

### URLs d'accès (localhost)

| Service | URL |
|---------|-----|
| Interface RecoFilm | http://localhost:8501 |
| MLflow | http://localhost:5000 |
| Airflow | http://localhost:8085 |
| Trainer API (Swagger) | http://localhost:8000/docs |
| Predicter API (Swagger) | http://localhost:8001/docs |
| KNN API (Swagger) | http://localhost:8002/docs |

---

## Prérequis

- **Docker** et **Docker Compose**
- Fichier **`.env`** à la racine du projet (voir section [Variables d'environnement](#variables-denvironnement))
- Compte **Kaggle** (pour le téléchargement des posters de films, optionnel)

---

## Installation et démarrage

### 1. Cloner le dépôt

```bash
git clone <url-du-repo>
cd sep25_cmlops_reco_films2
```

### 2. Configurer les variables d'environnement

Créer un fichier `.env` à la racine du projet (voir section dédiée ci-dessous).

### 3. Lancer tous les services

```bash
docker compose up -d
```

### 4. Attendre le démarrage

- **MLflow** : ~30 s  
- **Trainer / Predicter / KNN API** : ~1 min  
- **Streamlit** : ~30 s  
- **Airflow** : ~2 min (initialisation de la base PostgreSQL)

### 5. Accéder à l'application

- **Interface utilisateur** : http://localhost:8501  
- **Identifiants de démo** : `admin` / `RecoFilm!2025`

---

## Fonctionnement

### Flux utilisateur (Streamlit)

1. L'utilisateur se connecte avec `admin` / `RecoFilm!2025`
2. L'API KNN authentifie et retourne un token JWT + `user_id`
3. L'interface appelle `POST /predict` sur l'API KNN avec le token
4. Les recommandations (Top-N films avec titres, genres, notes) sont affichées avec pagination
5. Les posters sont chargés depuis Kaggle (dataset MovieLens 20M posters) si disponibles

### Flux d'entraînement (Airflow)

Le DAG `movie_training_pipeline` s'exécute **quotidiennement** (`@daily`) :

1. **insert_data** : insère les nouvelles données (ratings, tags, genome_scores) via l'API Trainer
2. **trigger_training** : entraîne le modèle SVD via l'API Trainer
3. **trigger_training_knn** : entraîne le modèle KNN via l'API KNN (authentification JWT)

Les tâches SVD et KNN s'exécutent en **parallèle** après l'insertion des données.

### APIs principales

| API | Endpoint | Méthode | Description |
|-----|----------|---------|-------------|
| Trainer | `/training` | POST | Entraîne le modèle SVD |
| Trainer | `/insert-data` | POST | Insère les données dans PostgreSQL |
| Trainer | `/health` | GET | Health check |
| Predicter | `/predict` | POST | Recommandations SVD |
| Predicter | `/reload-model` | POST | Recharge le modèle depuis MLflow |
| KNN | `/token` | POST | Authentification (username/password → JWT) |
| KNN | `/predict` | POST | Recommandations KNN (Bearer token requis) |
| KNN | `/training` | POST | Entraîne le modèle KNN (Bearer token requis) |

---

## Variables d'environnement

Créer un fichier `.env` à la racine du projet :

```env
# MLflow
MLFLOW_TRACKING_URI=http://mlflow:5000

# PostgreSQL (Supabase)
DATABASE_URL=postgresql://user:password@host:5432/postgres
DB_HOST=votre-host.supabase.com
DB_NAME=postgres
DB_USER=votre-user
DB_PASSWORD=votre-password
DB_PORT=5432
```

Pour **Airflow** (fichier `airflow/.env`) :

```env
API_KNN_TOKEN=votre-token-pour-api-knn
```

Les services `trainer`, `predicter`, `knn_api` et `mlflow` utilisent le `.env` racine via `env_file: .env` dans le `docker-compose.yml`.

---

## Structure du projet

```
sep25_cmlops_reco_films2/
├── airflow/                 # DAGs Airflow
│   ├── dags/
│   │   └── movie_training_pipeline.py
│   ├── airflow.cfg
│   └── .env
├── mlflow/                  # Service MLflow
├── trainer/                 # API d'entraînement SVD + insert-data
├── predicter/               # API de prédiction SVD
├── knn_api/                 # API KNN (training + predict + auth)
├── streamlit-ui/             # Interface Streamlit
│   ├── app.py
│   ├── demo.py
│   └── assets/
├── shared/                  # Code partagé (ex: svd_wrapper)
├── models/                  # Modèles entraînés (volumes Docker)
├── mlflow_data/             # Données MLflow (SQLite)
├── docker-compose.yml
└── .env
```

---

## Équipe

Jimmy Seyer / Yacine Madi Said / Jingzi Zhao / Monitoring : Nicolas

---

## Licence

Voir le fichier `LICENSE` du projet.
