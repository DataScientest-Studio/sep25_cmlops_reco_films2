"""
Sprint 1 - Fondations
"""

import streamlit as st

st.set_page_config(
    page_title="Sprint 1 - Fondations",
    page_icon="📊",
    layout="wide"
)

st.title(" Sprint 1 - Fondations")
st.markdown("**Deadline : 3 Novembre 2025**")

st.markdown("---")

# Objectifs
st.markdown("## Objectifs du Sprint")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Objectifs Principaux
    -  Définir les objectifs et la roadmap
    -  Environnement reproductible
    -  Collecter et prétraiter les données
    -  Base de données PostgreSQL
    -  Modèle ML baseline (KNN)
    -  API d'inférence (FastAPI)
    """)

with col2:
    st.markdown("""
    ### Livrables
    -  Repository Git structuré
    -  Base de données opérationnelle
    -  Modèle KNN entraîné
    -  API avec /training et /predict
    -  Documentation
    """)

# Dataset
st.markdown("---")
st.markdown("## Dataset MovieLens 20M")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Films", "27,278")
with col2:
    st.metric("Utilisateurs", "138,493")
with col3:
    st.metric("Ratings", "10M")

st.markdown("""
**Source :** [GroupLens MovieLens 20M](https://grouplens.org/datasets/movielens/20m/)

**Échantillonnage :**
- Dataset original : 20M ratings (~500 MB)
- Dataset utilisé : 10M ratings (~250 MB)
- **Raison :** Limite gratuite Supabase (500 MB)
- **Impact :** Négligeable (99% de couverture)
""")

# Base de données
st.markdown("---")
st.markdown("##  Base de Données PostgreSQL")

st.markdown("""
### Choix technique : PostgreSQL (Supabase Cloud)

**Pourquoi PostgreSQL plutôt que SQLite ?**
-  Scalabilité multi-utilisateurs
-  Production-ready
-  Meilleures performances
-  Cloud-native

### Tables créées :
- `movies` : Films avec titres et genres
- `ratings` : Notes des utilisateurs
- `tags` : Tags descriptifs
- `links` : Liens IMDb et TMDb
""")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Movies", "27,278")
with col2:
    st.metric("Ratings", "10,000,000")
with col3:
    st.metric("Tags", "465,564")
with col4:
    st.metric("Links", "27,278")

# Modèle
st.markdown("---")
st.markdown("##  Modèle KNN")

st.markdown("""
### K-Nearest Neighbors - Collaborative Filtering

**Configuration :**
- K = 20 voisins
- Algorithme : ball_tree
- Métrique : euclidienne

**Pourquoi KNN ?**
-  Simple et interprétable
-  Performant pour la recommandation
-  Gère bien la sparsité
""")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("K (voisins)", "20")
with col2:
    st.metric("Algorithme", "ball_tree")
with col3:
    st.metric("Temps d'entraînement", "< 1 sec")

# API
st.markdown("---")
st.markdown("## 🔌 API FastAPI")

st.markdown("""
### Endpoints disponibles :

**GET /** - Page d'accueil

**GET /health** - Health check
```json
{
    "status": "healthy",
    "model_loaded": true,
    "database_connected": true
}
```

**POST /training** - Entraînement du modèle

**POST /predict** - Recommandations
```json
{
    "userId": 1,
    "numRecommendations": 5
}
```
""")

# Défis
st.markdown("---")
st.markdown("## Défis & Solutions")

challenges = [
    ("Volumétrie", "20M ratings > 500 MB", "Échantillonnage à 10M"),
    ("Migration SQL", "Incompatibilité SQLite/PostgreSQL", "cursor.execute()"),
    ("Performance", "Preprocessing lent", "Traitement par batch")
]

for title, problem, solution in challenges:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"** {title}**")
        st.info(problem)
    with col2:
        st.markdown("** Solution**")
        st.success(solution)

# Footer
st.markdown("---")
st.success("Sprint 1 complété avec succès !")