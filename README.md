# RecoFilm - Système de Recommandation de Films MLOps

Projet MLOps de système de recommandation de films basé sur le dataset MovieLens 20M.

[![CI Pipeline](https://github.com/DataScientest-Studio/sep25_cmlops_reco_films2/actions/workflows/ci.yaml/badge.svg)](https://github.com/DataScientest-Studio/sep25_cmlops_reco_films2/actions)

---

## Table des matières

- [Présentation](#-présentation)
- [Architecture](#-architecture)
- [Technologies](#-technologies-utilisées)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Structure du projet](#-structure-du-projet)
- [Monitoring](#-monitoring)
- [CI/CD](#-cicd)
- [Équipe](#-équipe)

---

## Présentation

RecoFilm est un système de recommandation de films complet implémentant les meilleures pratiques MLOps:

- **API REST** FastAPI pour les prédictions en temps réel
- **Monitoring** avec Prometheus & Grafana
- **Détection de drift** automatique avec Evidently
- **Réentraînement automatique** du modèle
- **CI/CD** avec GitHub Actions
- **Containerisation** Docker complète

### Objectifs MLOps

- Déploiement automatisé
- Monitoring en temps réel
- Détection et correction du data drift
- Versioning des modèles avec MLflow
- Tests automatisés
- Documentation complète

---

## Architecture
```
┌─────────────────┐
│   Utilisateur   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│   API FastAPI   │◄────►│  PostgreSQL  │
│   (Port 8000)   │      │  (Supabase)  │
└────────┬────────┘      └──────────────┘
         │
         ├──────────────────┐
         │                  │
         ▼                  ▼
┌─────────────────┐ ┌─────────────────┐
│   Prometheus    │ │     MLflow      │
│   (Port 9090)   │ │   (Port 5000)   │
└────────┬────────┘ └─────────────────┘
         │
         ▼
┌─────────────────┐
│     Grafana     │
│   (Port 3000)   │
└─────────────────┘
```

### Workflow MLOps
```
Données ──► Preprocessing ──► Training ──► Model Registry
                                 │              │
                                 ▼              ▼
                            Evaluation ──► Deployment (API)
                                               │
                                               ▼
                                          Monitoring
                                               │
                                               ▼
                                      Drift Detection
                                               │
                                               ▼
                                    Auto-Retraining ──┐
                                               │      │
                                               └──────┘
```

---

## 🛠️ Technologies utilisées

### Backend & API
- **Python 3.13**
- **FastAPI** - API REST
- **PostgreSQL** (Supabase) - Base de données
- **scikit-learn** - Modèle KNN

### MLOps
- **MLflow** - Tracking & Model Registry
- **Evidently** - Détection de data drift
- **APScheduler** - Planification auto-retrain

### Monitoring
- **Prometheus** - Collecte de métriques
- **Grafana** - Visualisation

### DevOps
- **Docker & Docker Compose** - Containerisation
- **GitHub Actions** - CI/CD
- **pytest** - Tests automatisés

---

## Installation

### Prérequis

- Python 3.13+
- Docker & Docker Compose
- Git

### 1. Cloner le repository
```bash
git clone https://github.com/DataScientest-Studio/sep25_cmlops_reco_films2.git
cd sep25_cmlops_reco_films2
```

### 2. Configuration environnement

Créer un fichier `.env` à la racine:
```env
DB_HOST=db.mmjschyvinnqscoeijhn.supabase.co
DB_PORT=5432
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=votre_password
```

### 3. Lancer avec Docker
```bash
# Build et démarrage
docker-compose up -d

# Vérifier que tout tourne
docker-compose ps
```

### 4. Installation locale (optionnel)
```bash
# Créer environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installer dépendances
pip install -r requirements.txt
```

---

## Utilisation

### API FastAPI

**Swagger UI:**
```
http://localhost:8000/docs
```

**Endpoints principaux:**
```bash
# Health check
GET http://localhost:8000/health

# Prédictions
POST http://localhost:8000/predict
{
  "userId": 1,
  "numRecommendations": 10
}

# Entraînement
POST http://localhost:8000/training
```

**Exemple avec curl:**
```bash
# Test predict
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"userId": 1, "numRecommendations": 10}'
```

### Monitoring

**Grafana:**
```
http://localhost:3000
Login: admin / admin
```

**Dashboards disponibles:**
- Nombre de requêtes actives
- Total des requêtes HTTP
- Erreurs HTTP (4xx, 5xx)
- Temps de réponse moyen

**Prometheus:**
```
http://localhost:9090
```

**MLflow:**
```
http://localhost:5000
```

### Détection de Drift

**Lancer la détection manuellement:**
```bash
python src/monitoring/drift_detection.py
```

**Rapports générés dans:** `reports/drift/`

**Ouvrir les rapports HTML dans le navigateur**

### Réentraînement Automatique

**Lancer le réentraînement (si drift détecté):**
```bash
python src/monitoring/auto_retrain.py
```

**Planification automatique (tous les jours à 2h):**
```bash
python src/monitoring/schedule_retrain.py
```

**Logs de décision:** `logs/retrain/`

---

## Structure du projet
```
sep25_cmlops_reco_films2/
│
├── .github/
│   └── workflows/          # CI/CD pipelines
│       ├── ci.yaml         # Tests + Build
│       └── release.yaml    # Déploiement
│
├── data/
│   ├── raw/                # Données brutes MovieLens
│   └── processed/          # Matrices preprocessées
│
├── database/
│   ├── config.py           # Configuration PostgreSQL
│   └── init_db_postgres.py # Initialisation DB
│
├── grafana/
│   └── provisioning/       # Config Grafana (persistence)
│       ├── datasources/    # Prometheus datasource
│       └── dashboards/     # Dashboards JSON
│
├── logs/
│   └── retrain/            # Logs réentraînement auto
│
├── models/                 # Modèles ML sauvegardés
│   ├── model.pkl
│   └── movie_ids.pkl
│
├── reports/
│   └── drift/              # Rapports Evidently HTML
│
├── src/
│   ├── api/
│   │   └── main.py         # API FastAPI
│   ├── data/
│   │   └── preprocess.py   # Preprocessing données
│   ├── models/
│   │   ├── train_model.py  # Script entraînement
│   │   └── predict_model.py # Script prédiction
│   └── monitoring/
│       ├── drift_detection.py    # Détection drift
│       ├── auto_retrain.py       # Réentraînement auto
│       └── schedule_retrain.py   # Planification
│
├── tests/                  # Tests unitaires
│
├── docker-compose.yml      # Orchestration services
├── Dockerfile              # Image API
├── Dockerfile.mlflow       # Image MLflow
├── prometheus.yml          # Config Prometheus
├── requirements.txt        # Dépendances Python
└── README.md
```

---

## Monitoring

### Métriques collectées

**API Metrics:**
- Nombre total de requêtes
- Requêtes par seconde
- Temps de réponse
- Taux d'erreur (4xx, 5xx)
- Requêtes actives

**Data Drift Metrics:**
- Score de drift par feature
- Distribution des données
- Qualité des données

### Dashboards Grafana

**4 panels principaux:**
1. **Requêtes actives** - Surveillance en temps réel
2. **Total requêtes HTTP** - Volume cumulé
3. **Erreurs HTTP** - Détection anomalies
4. **Temps de réponse** - Performance API

---

## Tests

### Tests unitaires
```bash
# Lancer tous les tests
pytest

# Tests avec coverage
pytest --cov=src

# Tests d'un module spécifique
pytest tests/test_api.py
```

### Tests d'intégration
```bash
# Test API
curl http://localhost:8000/health

# Test prédiction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"userId": 1, "numRecommendations": 10}'
```

---

## CI/CD

### Continuous Integration (CI)

**Déclenché sur:** Chaque push

**Pipeline:**
1. Linter (code quality)
2. Tests unitaires
3. Build images Docker

**Fichier:** `.github/workflows/ci.yaml`

### Continuous Deployment (CD)

**Déclenché sur:** Releases/Tags

**Pipeline:**
1. Linter
2. Tests
3. Build images
4. Push sur DockerHub (si configuré)

**Fichier:** `.github/workflows/release.yaml`

### Créer une release
```bash
# Via Git
git tag v1.0.0
git push origin v1.0.0

# Ou via GitHub UI
# Releases → Create new release
```

---

## Équipe

**Formation:** DataScientest - MLOps Engineer

**Projet:** RecoFilm - Système de recommandation MLOps

**Mentor:** Nicolas

---

## 🔗 Liens utiles

- [Documentation FastAPI](https://fastapi.tiangolo.com/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Evidently Documentation](https://docs.evidentlyai.com/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)

---

**Développé dans le cadre de la formation MLOps DataScientest**