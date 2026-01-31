"""
Sprint 5 - Frontend (Streamlit)
"""

import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Sprint 5 - Frontend",
    page_icon="🎨",
    layout="wide"
)

st.title("🎨 Sprint 5 — Frontend (Streamlit)")
st.markdown("**UI démo pour le jury : user_id → appel API → recommandations**")

st.markdown("---")

# Indicateurs en haut de page
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.markdown("### 👤 Entrée")
    st.metric("user_id", "Input")

with col2:
    st.markdown("### 🔌 Call API")
    st.metric("/predict", "POST")

with col3:
    st.markdown("### 📤 Sortie")
    st.metric("Top-N", "Films")

with col4:
    st.markdown("### 🎬 Option")
    st.metric("posters", "TMDb")

with col5:
    st.markdown("### 🔧 Option")
    st.metric("version", "modèle")

with col6:
    st.markdown("### ✅ Démo")
    st.metric("30 sec", "Live")

st.markdown("---")


# Onglets principaux
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Objectifs",
    "🎬 Démo", 
    "🧩 Widgets",
    "📸 Captures",
    "⚡ Défis"
])

# =============================================
# TAB 1: OBJECTIFS
# =============================================
with tab1:
    st.markdown("## Objectifs (Sprint 5)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Objectifs Principaux
        - ✅ Rendre l'API utilisable par un non-tech
        - ✅ Démo claire : navigation + preuves (captures)
        - ✅ Uniformiser la charte visuelle
        """)
    
    with col2:
        st.markdown("""
        ### Livrables (preuves)
        - ✅ Application Streamlit fonctionnelle
        - ✅ Captures d'écran de chaque étape
        - ✅ Navigation fluide entre sprints
        - ✅ Démo live devant le jury
        """)

# =============================================
# TAB 2: DÉMO
# =============================================
with tab2:
    st.markdown("## 🎬 Démo (30 sec)")
    
    st.info("💡 **Capture à ajouter :** Écran démo user_id → recommandations")
    
    st.markdown("""
    ### Scénario de démonstration
    
    **Déroulé (30 secondes) :**
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        1. **Choisir un user_id**
        2. **Cliquer "Recommander"**
        3. **Afficher Top-N** (+ option poster)
        4. **Montrer /docs** en backup
        """)
    
    with col2:
        st.code("""
# Appel API sous le capot
POST http://localhost:8000/predict
{
  "userId": 1,
  "numRecommendations": 10
}

# Réponse JSON
{
  "userId": 1,
  "recommendations": [
    {
      "movieId": 5496,
      "title": "Ossessione (1943)",
      "genres": "Drama|Romance",
      "avg_rating": 3.85,
      "num_ratings": 134
    },
    ...
  ]
}
        """, language="json")

# =============================================
# TAB 3: WIDGETS
# =============================================
with tab3:
    st.markdown("## 🧩 Widgets recommandés")
    
    st.markdown("""
    Liste des widgets Streamlit utilisés dans l'application :
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Widgets de base
        - **`st.selectbox`** pour user_id
        - **`st.button`** pour déclencher /predict
        - **`st.dataframe`** pour afficher le Top-N
        - **`st.metric`** pour infos modèle (option)
        """)
        
        st.code("""
user_id = st.selectbox(
    "Choisir un utilisateur",
    options=[1, 2, 5, 10, 100]
)

if st.button("🎬 Recommander"):
    # Appel API
    recommendations = call_api(user_id)
    st.dataframe(recommendations)
        """, language="python")
    
    with col2:
        st.markdown("""
        ### Widgets avancés (option)
        - **`st.image`** pour afficher les posters
        - **`st.columns`** pour layout responsive
        - **`st.tabs`** pour organisation du contenu
        - **`st.expander`** pour détails techniques
        """)
        
        st.code("""
# Afficher posters (option)
for movie in recommendations:
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image(poster_url)
    with col2:
        st.write(movie['title'])
        """, language="python")

# =============================================
# TAB 4: CAPTURES
# =============================================
with tab4:
    st.markdown("## 📸 Captures")
    
    st.markdown("### Swagger UI - Documentation API")
    swagger_path = Path(__file__).parent.parent / "captures" / "swagger_ui.png"
    st.image(str(swagger_path), caption="Interface Swagger - RecoFilm API v2.0.0", width=1200)    
    
    st.markdown("""
    ### Endpoints disponibles
    
    Notre API expose **5 endpoints principaux** :
    
    - **GET /metrics** - Métriques Prometheus
    - **GET /** - Page d'accueil API
    - **GET /health** - Health check (status, model loaded, DB connected)
    - **POST /training** - Entraînement du modèle
    - **POST /predict** - Recommandations de films
    """)
    
    st.markdown("---")
    st.markdown("### Organisation des fichiers")
    
    st.code("""
# Structure recommandée
captures/
  ├─ grafana_dashboard.png ✅
  ├─ evidently_drift.png ✅
  └─ swagger_ui.png ✅
    """, language="bash")

# =============================================
# TAB 5: DÉFIS
# =============================================
with tab5:
    st.markdown("## ⚡ Défis & Solutions")
    
    challenges = [
        {
            "title": "🎨 Design cohérent",
            "problem": "Faire une UI pro sans designer",
            "solution": "Utiliser les widgets Streamlit natifs + palette de couleurs simple (DataScientest)"
        },
        {
            "title": "⚡ Performance",
            "problem": "Temps de réponse API > 1 sec avec posters",
            "solution": "Cache TMDb + limiter à 5-10 films affichés"
        },
        {
            "title": "🛠️ Démo live",
            "problem": "Que faire si l'API crash pendant la soutenance?",
            "solution": "Swagger UI en backup + captures d'écran préparées"
        }
    ]
    
    for challenge in challenges:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"### {challenge['title']}")
            st.info(f"**Problème :** {challenge['problem']}")
        with col2:
            st.markdown("### ✅ Solution")
            st.success(challenge['solution'])
        st.markdown("---")

# Footer
st.markdown("---")
