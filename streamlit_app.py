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
st.title("RecoFilm")
st.subheader("Système de Recommandation de Films - Architecture MLOps Complète")

st.markdown("---")

# Introduction
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## Présentation du Projet")
    
    st.markdown("""
    ### Contexte
    
    Ce projet s'inscrit dans le cadre de la **formation MLOps** de DataScientest et vise à concevoir 
    et déployer une architecture MLOps complète pour un système de recommandation de films.
    
    ### Objectifs
    
    -  Construire un système de recommandation performant basé sur le dataset MovieLens 20M
    -  Mettre en place une pipeline MLOps complète (versioning, tracking, déploiement, monitoring)
    -  Déployer l'application avec une architecture scalable et maintenable
    -  Assurer le monitoring et la maintenance du système en production
    """)

with col2:
    st.markdown("## Métriques Clés")
    
    st.metric("Films", "27,278")
    st.metric("Utilisateurs", "138,493")
    st.metric("Ratings", "10M")
    st.metric("Modèle", "KNN")

# Technologies
st.markdown("---")
st.markdown("## Technologies Utilisées")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### Données
    - PostgreSQL (Supabase)
    - Pandas / NumPy
    """)

with col2:
    st.markdown("""
    ### ML & MLOps
    - Scikit-learn (KNN)
    - MLflow
    - FastAPI
    """)

with col3:
    st.markdown("""
    ### Déploiement
    - Docker
    - GitHub Actions
    - Streamlit
    """)

# Sidebar
with st.sidebar:
    st.markdown("## Navigation")
    st.info("Utilisez le menu ci-dessus pour naviguer entre les sprints")
    
    st.markdown("---")
    st.markdown("## Projet")
    st.markdown("**Formation :** MLOps - DataScientest")
    st.markdown("**Période :** Sept 2025 - Fév 2026")
    st.markdown("**Soutenance :** 23 Février 2026")