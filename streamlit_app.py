"""
RecoFilm - Système de Recommandation MLOps
Application Streamlit pour la présentation de soutenance
"""

import streamlit as st

# Configuration de la page
st.set_page_config(
    page_title="RecoFilm - MLOps Project",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header
st.title("🎬 RecoFilm")
st.subheader("Système de Recommandation de Films - Architecture MLOps Complète")

st.markdown("---")

# Introduction
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## 📋 Présentation du Projet")
    
    st.markdown("""
    ### Contexte
    
    Ce projet s'inscrit dans le cadre de la **formation MLOps** de DataScientest et vise à concevoir 
    et déployer une architecture MLOps complète pour un système de recommandation de films.
    
    ### Objectifs
    
    - 🎯 Construire un système de recommandation performant basé sur le dataset MovieLens 20M
    - 🔧 Mettre en place une pipeline MLOps complète (versioning, tracking, déploiement, monitoring)
    - 🚀 Déployer l'application avec une architecture scalable et maintenable
    - 📊 Assurer le monitoring et la maintenance du système en production
    """)

with col2:
    st.markdown("## 📊 Métriques Clés")
    
    st.metric("Films", "27,278")
    st.metric("Utilisateurs", "138,493")
    st.metric("Ratings", "10M")
    st.metric("Modèle", "KNN")

# Architecture
st.markdown("---")
st.markdown("## 🏗️ Architecture MLOps")

st.markdown("""
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
""")

# Technologies
st.markdown("---")
st.markdown("## 🛠️ Technologies Utilisées")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### Données
    - PostgreSQL (Supabase)
    - Pandas / NumPy
    - MovieLens 20M
    """)

with col2:
    st.markdown("""
    ### ML & MLOps
    - Scikit-learn (KNN)
    - MLflow (Tracking)
    - FastAPI
    - Evidently (Drift)
    """)

with col3:
    st.markdown("""
    ### Déploiement & Monitoring
    - Docker / Docker Compose
    - Prometheus / Grafana
    - GitHub Actions (CI/CD)
    - Streamlit
    """)

# Sprints
st.markdown("---")
st.markdown("## 🎯 Organisation par Sprints")

sprint_info = [
    {
        "title": "Sprint 1 - Fondations",
        "icon": "📊",
        "description": "Setup environnement, données, modèle baseline, API",
        "deadline": "3 Novembre 2025"
    },
    {
        "title": "Sprint 2 - Microservices & Suivi",
        "icon": "🔄",
        "description": "MLflow tracking, versioning données/modèles",
        "deadline": "5 Décembre 2025"
    },
    {
        "title": "Sprint 3 - Orchestration & Déploiement",
        "icon": "🐳",
        "description": "Docker, CI/CD, tests unitaires",
        "deadline": "2 Janvier 2026"
    },
    {
        "title": "Sprint 4 - Monitoring & Maintenance",
        "icon": "📊",
        "description": "Grafana/Prometheus, Evidently, auto-retrain",
        "deadline": "30 Janvier 2026"
    },
    {
        "title": "Sprint 5 - Frontend",
        "icon": "🎨",
        "description": "Interface Streamlit pour démo jury",
        "deadline": "20 Février 2026"
    }
]

for sprint in sprint_info:
    with st.expander(f"{sprint['icon']} {sprint['title']}"):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**Description :** {sprint['description']}")
        with col2:
            st.markdown(f"**Deadline :** {sprint['deadline']}")

# Équipe
st.markdown("---")
st.markdown("## 👥 Équipe")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### Jimmy
    **Sprint 4 & 5**
    - Monitoring & Maintenance
    - Frontend Streamlit
    """)

with col2:
    st.markdown("""
    ### Yacine
    **Sprint 2 & 3**
    - Microservices & Versioning
    - Orchestration & Déploiement
    """)

with col3:
    st.markdown("""
    ### Équipe
    **Sprint 0 & 1**
    - Introduction & Fondations
    - Setup projet
    """)

# Sidebar
with st.sidebar:
    st.markdown("## 🧭 Navigation")
    st.info("Utilisez le menu ci-dessus pour naviguer entre les sprints")
    
    st.markdown("---")
    st.markdown("## 📅 Projet")
    st.markdown("**Formation :** MLOps - DataScientest")
    st.markdown("**Période :** Sept 2025 - Fév 2026")
    st.markdown("**Soutenance :** 23 Février 2026")
    
    st.markdown("---")
    st.markdown("## 🔗 Liens Utiles")
    st.markdown("- [API Swagger](http://localhost:8000/docs)")
    st.markdown("- [Grafana](http://localhost:3000)")
    st.markdown("- [Prometheus](http://localhost:9090)")
    st.markdown("- [MLflow](http://localhost:5000)")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>RecoFilm</strong> - Projet MLOps DataScientest 2025-2026</p>
    <p><em>Système de recommandation de films avec architecture MLOps complète</em></p>
</div>
""", unsafe_allow_html=True)