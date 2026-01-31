import streamlit as st
import requests

# Configuration
FASTAPI_URL = "http://movie_trainer_api:8000"

st.title("🔐 Authentification - RecoFilm")
st.markdown("Connecte-toi pour accéder à l'API de recommandation.")

# Si l'utilisateur est connecté, affiche les options
if "token" in st.session_state:
    st.success(f"Tu es connecté en tant que {st.session_state['username']}.")

    # Section pour appeler l'API protégée
    st.header("🚀 Appeler l'API")

    if st.button("Lancer l'entraînement du modèle"):
        headers = {"Authorization": f"Bearer {st.session_state['token']}"}
        response = requests.post(f"{FASTAPI_URL}/training", headers=headers)

        if response.status_code == 200:
            result = response.json()
            st.json(result)
        else:
            st.error(f"Erreur : {response.text}")

    if st.button("Se déconnecter"):
        del st.session_state["token"]
        st.rerun()
else:
    st.warning("Veuillez vous connecter pour accéder aux fonctionnalités.")
    # Formulaire de connexion
    with st.form("login_form"):
        username = st.text_input("Nom d'utilisateur", value="admin")
        password = st.text_input("Mot de passe", type="password", value="secret")
        submitted = st.form_submit_button("Se connecter")

        if submitted:
            # Requête pour obtenir un token JWT
            response = requests.post(
                f"{FASTAPI_URL}/token",
                data={"username": username, "password": password}
            )

            if response.status_code == 200:
                token = response.json()["access_token"]
                print(token)
                st.session_state["token"] = token
                st.session_state["username"] = username
                st.success("Connexion réussie !")
                st.rerun()
            else:
                st.error("Nom d'utilisateur ou mot de passe incorrect.")