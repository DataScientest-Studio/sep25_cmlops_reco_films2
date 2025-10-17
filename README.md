# Projet MLOps - Systeme de Recommandation de Films (RecoFilm)

Projet de systeme de recommandation de films utilisant MovieLens 20M, dans le cadre du cursus MLOps DataScientest.

## Description

Ce projet implemente un systeme de recommandation de films utilisant :
- **Collaborative Filtering** avec K-Nearest Neighbors
- **Content-Based Filtering** base sur les genres et metadonnees
- Architecture MLOps complete avec versioning, CI/CD, monitoring et API

**Donnees** : MovieLens 20M (20 millions de notes, 27k films, 138k utilisateurs)

---

## Setup du projet

### 1. Cloner le repository

```bash
git clone https://github.com/DataScientest-Studio/sep25_cmlops_reco_films2.git
cd sep25_cmlops_reco_films2
```

### 2. Creer l'environnement virtuel

```bash
python -m venv venv
```

### 3. Activer l'environnement virtuel

**Windows (Command Prompt)** :
```bash
venv\Scripts\activate.bat
```

**Windows (PowerShell)** :
```bash
venv\Scripts\activate
```

**Linux/Mac** :
```bash
source venv/bin/activate
```

### 4. Installer les dependances

```bash
pip install -r requirements.txt
```

---

## Workflow de preparation des donnees

### 1. Telecharger les donnees MovieLens 20M

```bash
python src/data/download_data.py
```

Les donnees seront telechargees dans `data/raw/ml-20m/`.

### 2. Creer la base de donnees SQLite

```bash
python database/init_db.py
```

Cela creera `database/recofilm.db` avec les tables necessaires.

### 3. Inserer les donnees dans la base

```bash
python src/data/ingest_data.py
```

Cette etape prend environ 3-5 minutes (20M de ratings a inserer).

### 4. (Optionnel) Explorer les donnees

Lancer Jupyter Notebook :
```bash
jupyter notebook
```

Ouvrir `notebooks/exploration.ipynb` pour visualiser les statistiques des donnees.

---

## Structure du projet

```
├── database/              <- Base de donnees SQLite
│   ├── init_db.py        <- Script de creation des tables
│   └── recofilm.db       <- Base de donnees (non versionne)
│
├── data/
│   ├── raw/              <- Donnees brutes MovieLens (non versionne)
│   └── processed/        <- Donnees pretraitees (matrices)
│
├── models/               <- Modeles entraines (non versionne)
│
├── notebooks/            <- Notebooks Jupyter d'exploration
│   └── exploration.ipynb
│
├── src/
│   ├── data/            <- Scripts de gestion des donnees
│   │   ├── download_data.py   <- Telechargement MovieLens
│   │   ├── ingest_data.py     <- Ingestion en BDD
│   │   └── preprocess.py      <- Preprocessing (a venir)
│   │
│   ├── models/          <- Scripts ML
│   │   ├── train_model.py     <- Entrainement (a venir)
│   │   └── predict_model.py   <- Predictions (a venir)
│   │
│   └── api/             <- API FastAPI (a venir)
│       └── main.py
│
├── requirements.txt     <- Dependances Python
├── .gitignore          <- Fichiers a ignorer (donnees, venv, etc.)
└── README.md           <- Ce fichier
```

---

## Phases du projet

### Phase 1 : Fondations (deadline : 3 novembre)
- [x] Setup environnement
- [x] Telechargement des donnees
- [x] Exploration des donnees
- [x] Base de donnees SQLite
- [ ] Preprocessing (matrices)
- [ ] Modele ML de base
- [ ] API d'inference simple

### Phase 2 : Suivi & Versionning (deadline : 5 decembre)
- [ ] MLflow pour le tracking
- [ ] Versionning des donnees/modeles
- [ ] Comparaison des experiences

### Phase 3 : Deploiement (deadline : 2 janvier)
- [ ] Dockerisation
- [ ] CI/CD avec GitHub Actions
- [ ] Tests unitaires

### Phase 4 : Monitoring (deadline : 30 janvier)
- [ ] Prometheus/Grafana
- [ ] Detection de data drift (Evidently)
- [ ] Re-entrainement automatique

### Phase 5 : Frontend (deadline : 20 fevrier)
- [ ] Application Streamlit

**Soutenance** : 23 fevrier

---

## Technologies utilisees

- **Python 3.13**
- **pandas, numpy** - Manipulation de donnees
- **scikit-learn** - Machine Learning
- **SQLite** - Base de donnees
- **FastAPI** - API REST
- **Jupyter** - Exploration de donnees
- **Git/GitHub** - Versionning

---

## Ressources

- [Dataset MovieLens 20M](https://grouplens.org/datasets/movielens/20m/)
- [Template GitHub](https://github.com/DataScientest-Studio/Template_MLOps_movie_recommendation)
- [Documentation projet](https://docs.google.com/presentation/d/1VENc3vnN3zI3Z4WsLV8Q_ePjRowdMU2bsB-ylyAQV60/)

---

## Notes importantes

- Les donnees (`data/raw/`), la base de donnees (`database/*.db`) et l'environnement virtuel (`venv/`) ne sont **pas versionnes** sur Git
- Chaque membre de l'equipe doit recreer ces elements localement avec les scripts fournis
- La taille de la BDD finale est d'environ **1.2 GB**

---

<p><small>Projet base sur le template <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science</a></small></p>