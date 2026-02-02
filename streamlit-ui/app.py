import streamlit as st
import requests
import os

# -----------------------------
# CONFIGURATION
# -----------------------------
FASTAPI_TRAINING_URL = "http://movie_trainer_api:8000"
FASTAPI_PREDICTION_URL = "http://movie_predicter_api:8000"
FASTAPI_KNN_URL = "http://knn_api:8000"
DEFAULT_RECO = 100  # Nombre par défaut de recommandations
MOVIES_PER_PAGE = 5  # Nombre de films par page


def get_recommendations(token: str, userid: int, num_recommendations: int = 10):
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"userid": userid, "numRecommendations": num_recommendations}
    
    response = requests.post(
        f"{FASTAPI_KNN_URL}/predict",
        headers=headers,
        json=payload
    )
    
    if response.status_code == 200:
        return response.json()["recommendations"]
    else:
        st.error(f"Erreur API: {response.status_code} {response.text}")
        return []
    
    
# -----------------------------
# SESSION STATE INITIALIZATION
# -----------------------------
if "token" not in st.session_state:
    st.session_state.token = None
if "username" not in st.session_state:
    st.session_state.username = None
if "userid" not in st.session_state:
    st.session_state.userid = None
if "recommendations" not in st.session_state:
    st.session_state.recommendations = []
if "index" not in st.session_state:
    st.session_state.index = 0
if "recommandations_ready" not in st.session_state:
    st.session_state.recommandations_ready = False

# -----------------------------
# TITLE
# -----------------------------
st.title("🔐 Recommandation de films")
# -----------------------------
# USER CONNECTED
# -----------------------------
if st.session_state.recommandations_ready and st.session_state.token:
    st.success(f"Connecté en tant que {st.session_state.username} (UserId: {st.session_state.userid})")

    # Slideshow horizontal
    recs = st.session_state.recommendations
    if recs:
        start_idx = st.session_state.index
        end_idx = start_idx + MOVIES_PER_PAGE
        page = recs[start_idx:end_idx]

        # Boutons Prev / Next
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            if st.button("⬅️ Prev"):
                st.session_state.index = (start_idx - MOVIES_PER_PAGE) % len(recs)
                st.rerun()
        with col3:
            if st.button("Next ➡️"):
                st.session_state.index = (start_idx + MOVIES_PER_PAGE) % len(recs)
                st.rerun()

        # Affichage horizontal
        movie_cols = st.columns(MOVIES_PER_PAGE)
        for idx, movie in enumerate(page):
            with movie_cols[idx]:
                st.caption(f"Movie ID: {movie['movieid']}")
                st.markdown(f"**{movie['title']}**")
                st.markdown(f"Genres: {movie['genres']}")
                st.markdown(f"⭐ Note moyenne: `{movie['avg_rating']:.2f}`")
                st.markdown(f"Note prédite: `{movie['svg_pred_rate']:.2f}`")

        st.caption(f"Page {start_idx // MOVIES_PER_PAGE + 1} / {(len(recs)-1)//MOVIES_PER_PAGE + 1}")
    else:
        st.info("Aucune recommandation. Cliquez sur 'Get recommendations' pour charger les films.")

    # Déconnexion
    if st.button("Se déconnecter"):
        for key in ["token", "username", "userid", "recommendations", "index","recommandations_ready"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# -----------------------------
# USER NOT CONNECTED
# -----------------------------
else:
    st.warning("Veuillez vous connecter pour accéder aux recommandations.")

    with st.form("login_form"):
        username = st.text_input("Nom d'utilisateur", value="admin")
        password = st.text_input("Mot de passe", type="password", value="RecoFilm!2025")
        submitted = st.form_submit_button("Se connecter")

        if submitted:
            with st.spinner("Connexion en cours et récupération des recommandations..."):
                # Login
                response = requests.post(
                    f"{FASTAPI_KNN_URL}/token",
                    data={"username": username, "password": password}
                )
                if response.status_code == 200:
                    token = response.json()["access_token"]
                    userid = response.json()["userid"]

                    # Requête à l'API KNN pour les recommandations
                    recommendations = get_recommendations(token, userid, DEFAULT_RECO)

                    if recommendations:
                        st.session_state.token = token
                        st.session_state.username = username
                        st.session_state.userid = userid
                        st.session_state.recommandations_ready = True
                        st.session_state.recommendations = recommendations
                        st.session_state.index = 0
                        st.success("Connexion réussie ! Recommandations chargées ✅")
                        st.rerun()
                    else:
                        st.session_state.token = token
                        st.session_state.username = username
                        st.session_state.userid = userid
                        st.session_state.recommandations_ready = True
                        st.session_state.recommendations = []
                        st.session_state.index = 0
                        st.error("Impossible de récupérer les recommandations.")
                        st.rerun()
                else:
                    st.error("Nom d'utilisateur ou mot de passe incorrect.")
