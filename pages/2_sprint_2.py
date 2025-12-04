"""
Sprint 2 - MLflow Tracking & Model Registry
"""

import streamlit as st

st.set_page_config(
    page_title="Sprint 2 - MLflow",
    page_icon="🔄",
    layout="wide"
)

st.title("Sprint 2 - MLflow Tracking & Model Registry")
st.markdown("**Deadline : 5 Décembre 2025**")

st.markdown("---")

# Objectifs
st.markdown("## Objectifs du Sprint")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Objectifs Principaux
    - Mettre en place MLflow Tracking
    - Logger paramètres, métriques et artifacts
    - Implémenter MLflow Model Registry
    - Versionner automatiquement les modèles
    - Comparer les performances
    - Promouvoir le meilleur modèle
    """)

with col2:
    st.markdown("""
    ### Livrables
    - MLflow UI opérationnel
    - Tracking automatique des runs
    - Model Registry avec 7+ versions
    - Système de promotion automatique
    - Comparaison visuelle des modèles
    """)

# MLflow Tracking
st.markdown("---")
st.markdown("## MLflow Tracking")

st.markdown("""
MLflow Tracking permet d'enregistrer automatiquement tous les entraînements avec leurs paramètres, 
métriques et artifacts.

### Implémentation

Chaque entraînement crée un **run** qui contient :
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    #### Paramètres
    - `n_neighbors` : K (nombre de voisins)
    - `algorithm` : ball_tree, brute, kd_tree
    - `metric` : euclidean, manhattan
    """)

with col2:
    st.markdown("""
    #### Métriques
    - `training_time_seconds`
    - `n_samples` : 27,278 films
    - `n_features` : 21
    - `model_size_kb`
    - `avg_test_distance` (qualité)
    """)

with col3:
    st.markdown("""
    #### Artifacts
    - `model.pkl` (modèle KNN)
    - `movie_ids.pkl` (mapping)
    - Modèle sklearn loggé
    """)

# Code exemple
st.markdown("### Code d'intégration")

st.code("""
import mlflow
import mlflow.sklearn

# Configuration
mlflow.set_tracking_uri("file:///./mlruns")
mlflow.set_experiment("recofilm-knn-recommender")

# Démarrer un run
with mlflow.start_run(run_name=f"knn-training-{timestamp}"):
    
    # Logger les paramètres
    mlflow.log_params(params)
    
    # Entraîner le modèle
    model = train_model(...)
    
    # Logger les métriques
    mlflow.log_metric("training_time", time)
    mlflow.log_metric("avg_test_distance", distance)
    
    # Logger le modèle
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_artifact("model.pkl")
""", language="python")

# Résultats
st.markdown("---")
st.markdown("## Résultats obtenus")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Runs enregistrés", "7+")
with col2:
    st.metric("Expériences", "1")
with col3:
    st.metric("Meilleur K", "20")
with col4:
    st.metric("Temps moyen", "0.05s")

st.info("""
** Insight :** Les algorithmes `ball_tree` et `brute` avec différentes métriques (euclidean vs manhattan) 
produisent des résultats similaires car la métrique de test utilise uniquement les 5 premiers voisins.
""")

# Model Registry
st.markdown("---")
st.markdown("## MLflow Model Registry")

st.markdown("""
Le Model Registry permet de :
- **Versionner** automatiquement chaque modèle entraîné
- **Comparer** les performances entre versions
- **Promouvoir** le meilleur modèle avec un alias

### Système d'Aliases (moderne)

Au lieu de l'ancien système de "Stages" (Production/Staging/Archive), MLflow 3.6.0 utilise les **Aliases** :
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    #### Alias "champion"
    - Attribué au **meilleur modèle** automatiquement
    - Basé sur la métrique `avg_test_distance`
    - Visible directement dans la liste des versions
    - Plus un modèle est proche (distance faible), meilleur il est
    """)

with col2:
    st.markdown("""
    #### Promotion automatique
    1. Nouveau modèle entraîné → Version créée
    2. Comparaison avec le champion actuel
    3. Si meilleur (distance < champion) :
       - Retrait de l'alias du champion actuel
       - Attribution de l'alias au nouveau modèle
    4. Sinon : Version reste sans alias
    """)

# Code Registry
st.markdown("### Code de promotion automatique")

st.code("""
def compare_and_promote(model_name, current_version, avg_distance):
    client = MlflowClient()
    
    try:
        # Récupérer le champion actuel
        champion = client.get_model_version_by_alias(model_name, "champion")
        champion_distance = get_metric(champion.run_id, "avg_test_distance")
        
        # Comparer
        if avg_distance < champion_distance:
            # Nouveau champion !
            client.delete_model_version_alias(model_name, "champion")
            client.set_registered_model_alias(model_name, "champion", current_version)
            print(f" Version {current_version} promue champion!")
        else:
            print(f" Champion actuel reste meilleur")
            
    except:
        # Aucun champion → Premier modèle
        client.set_registered_model_alias(model_name, "champion", current_version)
        print(f" Premier champion : version {current_version}")
""", language="python")

# Comparaison visuelle
st.markdown("---")
st.markdown("## Comparaison des modèles")

st.markdown("""
MLflow UI permet de comparer visuellement les runs avec plusieurs types de graphiques :

- **Parallel Coordinates Plot** : Voir la relation entre paramètres et métriques
- **Scatter Plot** : Comparer 2 métriques
- **Box Plot** : Distribution des métriques
- **Contour Plot** : Zones de performances optimales
""")

st.markdown("""
### Exemple de comparaison

Avec 7 versions testées, on peut observer que :
- **K=20 vs K=30** : Distance identique (1315.8) car métrique basée sur 5 premiers voisins
- **ball_tree vs brute** : Performances similaires, mais `brute` légèrement plus rapide (0.00s vs 0.05s)
- **euclidean vs manhattan** : Métriques différentes produisent des distances différentes
""")

# Architecture
st.markdown("---")
st.markdown("## Architecture MLflow")

st.code("""
Project/
├── mlruns/                    # Stockage local MLflow
│   ├── 0/                     # Expérience Default
│   ├── 1/                     # Expérience recofilm-knn-recommender
│   │   ├── [run_id_1]/        # Run 1
│   │   │   ├── artifacts/     # model.pkl, movie_ids.pkl
│   │   │   ├── metrics/       # training_time, avg_distance, etc.
│   │   │   └── params/        # n_neighbors, algorithm, metric
│   │   └── ...
│   └── models/                # Model Registry
│       └── recofilm-knn-recommender/
│           ├── version-1/
│           ├── version-2/
│           └── version-7/     # Champion 
└── src/models/train_model.py # Script d'entraînement
""", language="bash")

# Défis
st.markdown("---")
st.markdown("## Défis & Solutions")

challenges = [
    ("Stages deprecated", "MLflow 2.9.0+ utilise Stages (Production/Staging)", "Migration vers Aliases modernes"),
    ("Métriques identiques", "K=20 vs K=30 donnent même distance", "Ajout paramètre metric pour différencier"),
    ("Filesystem deprecated", "Warning sur stockage fichiers local", "À migrer vers SQLite en production"),
]

for title, problem, solution in challenges:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"** {title}**")
        st.info(problem)
    with col2:
        st.markdown("** Solution**")
        st.success(solution)

# Commandes utiles
st.markdown("---")
st.markdown("## Commandes utiles")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Entraînement")
    st.code("""
# Lancer un entraînement
python src/models/train_model.py

# Le script :
# 1. Charge les données
# 2. Entraîne le modèle KNN
# 3. Logs dans MLflow
# 4. Enregistre dans Registry
# 5. Compare et promeut si meilleur
    """, language="bash")

with col2:
    st.markdown("### Visualisation")
    st.code("""
# Lancer MLflow UI
mlflow ui

# Ouvrir le navigateur
http://localhost:5000

# Onglets disponibles :
# - Experiments : Tous les runs
# - Models : Registry + Aliases
    """, language="bash")

# Prochaines étapes
st.markdown("---")
st.markdown("## Prochaines étapes")

st.markdown("""
### Sprint 3 : Docker & CI/CD
- Conteneurisation de l'API
- Conteneurisation de MLflow
- Pipeline GitHub Actions
- Tests automatisés

### Sprint 4 : Monitoring
- Prometheus/Grafana pour métriques système
- Evidently pour détection de drift
- Alertes automatiques
""")

# Footer
st.markdown("---")
st.success("Sprint 2 complété avec succès ! MLflow opérationnel avec 7 versions et système de promotion automatique.")