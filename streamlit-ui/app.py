import streamlit as st
import requests
import os

# -----------------------------
# CONFIGURATION
# -----------------------------
FASTAPI_TRAINING_URL = "http://movie_trainer_api:8000"
FASTAPI_PREDICTION_URL = "http://movie_predicter_api:8000"
FASTAPI_KNN_URL = "http://knn_api:8000"

MOVIES_PER_PAGE = 5  # Nombre de films par page

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

# -----------------------------
# TITLE
# -----------------------------
st.title("🔐 Recommandation de films")
# -----------------------------
# USER CONNECTED
# -----------------------------
if st.session_state.token:
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
                
                st.caption(f"Movie ID: {movie['movieId']}")
                if "poster" in movie and movie["poster"]:
                    st.image(movie["poster"], width=200)
                st.markdown(f"{movie['title']}")
                st.markdown(f"⭐ **Predicted rating:** `{movie['score']:.2f}`")

        st.caption(f"Page {start_idx // MOVIES_PER_PAGE + 1} / {(len(recs)-1)//MOVIES_PER_PAGE + 1}")

    else:
        st.info("Aucune recommandation. Cliquez sur 'Get recommendations' pour charger les films.")

    # Déconnexion
    if st.button("Se déconnecter"):
        for key in ["token", "username", "userid", "recommendations", "index"]:
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
            response = requests.post(
                f"{FASTAPI_KNN_URL}/token",
                data={"username": username, "password": password}
            )

            # Simuler récupération de recommandations KNN
            api_response = {
                "recommendations": [
                    {"movieId": 1, "title": "Toy Story", "score": 4.82, "poster": "https://storage.googleapis.com/kagglesdsdata/datasets/38615/58801/MLP-20M/MLP-20M/1.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20260201%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260201T120713Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=a3d0a98219207b0a7f2593ad5700ff063cd20516b23ac588b0895d13060f23ff27d0e72591d5927fdb349da217890a6fc6e16cad5da228efa95bff0222b590ee7834bd33f858f430715ecef4e6c87f7094348444d320bec7346e8df666246bac1c861a14d3022aedf86af74ef75f382e6a1d5f89e296d794bd6a6cf1f969bccb194980806705ddde3ad700db421aefd248f2bff3d34cfd4a526d290e3b67b339352b49ee3f15d15d01bc4ee38c6c458ab4c1644ad6babf9889acda3e36f0d240e82c3b6d45a38543e963a43fadad8e27100e0663dcc4750a9baf8bf35cd914bcb4ab18f8b968ef34ad46579d3787523df704c69611d05e2fa95381de3ce763ce"},
                    {"movieId": 318, "title": "Shawshank Redemption", "score": 4.91, "poster": "https://storage.googleapis.com/kagglesdsdata/datasets/38615/58801/MLP-20M/MLP-20M/10.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20260201%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260201T120730Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=23bd11fa75bfe5f83e4d79d7f52a2cd2c18bc252b6107a9d75966034f0ee42d451a7909409315fcc3147528c69a00d90a3697c475d290e6f490a337cc52d80239212cd9fa1c41ab8b0faf67daed7cd875b137a2b09bc948601a5359a92d1a903709edc47793e08d8d7b2573263f0289ea96d39875b2be2855126e10e46e3b588d3077f880162450d12d8d826ded1f8c11031c40319c4eb874b15cc35b31f5a81c453893eaeace29aa50eb9e5693d84f63eb460f8a2b802e43646009fda8a8ccbf45fcd2323042654731d7018191a7f0f5f3f545bf073a1ccbe776bed7bfb5f0d75579a0ba4a226a6005eb1f2e9fb6654b4a68d2072daf196481558c036d8d728"},
                    {"movieId": 356, "title": "Forrest Gump", "score": 4.77, "poster": "https://storage.googleapis.com/kagglesdsdata/datasets/38615/58801/MLP-20M/MLP-20M/100.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20260201%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260201T120743Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=cab5f40336611a57e2cd25ff8dc0ec3a3a993e15f1e6185ef7c313739ff0751035db1ff679eea41dbc9502fd2515abe73bbb2c0c7323eaa4e58dc783590d0fd9075d2cac4d0fff3056e2f073a4734f9a7c65689fc56e511e4357827adb2cd7691a3d47e90ff5f6838643d3c4ebf1c0dea50a83fceccb617733579654c2e3f840fcc447de44b4c65598e09c83ad62e4024d5497e454255d4e03b45d5bd0784a3407c88271d7c2ed51b999a5cb75bc4e7f55a663697afb2a9213fe1fc15bba5904a872ef865756d5f8be7cf6cac3169a2de3e2fba85c41263fea823ab3bc08845316a9c874a29f0f382ca469aeadb8632cda545f82473d9a9000d80f1666b84320"},
                    {"movieId": 50, "title": "Star Wars", "score": 4.65, "poster": "https://storage.googleapis.com/kagglesdsdata/datasets/38615/58801/MLP-20M/MLP-20M/1000.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20260201%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260201T125122Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=94198471fcd094f1edc1d0e86d7a473909c6b5b1372e490bbc04049c9250ab8ecf0a51d132240db2e51d0572beefdc8a82e8c56121c0962ee5db9aeb8f4a6fea9f201d4e14317fa59b6cbce951a114099d2c91834ecd85ed0a22e01c1d3775514696a30652ae63a7f2b6e7c8aa0fd3270bb17b01e24f05fb3f17b7f9807ca86772db6340649edf46d2ed80f203de8185c10ff548025a0d368c4f87bf8a0de9e04b6822b545857be9e3b4fb1529b39c0482cef8338b15d826f1701c2d263ccb112006187f3a59f1c1ef7e9657b9557166a5b1b260fee7fc622741f2380b24a40fd767aa77dc22d1d2a409e8dc130828fb394878e8a29cb494b7587d0d09d8e633"},
                    {"movieId": 150, "title": "Fight Club", "score": 4.88, "poster": "https://storage.googleapis.com/kagglesdsdata/datasets/38615/58801/MLP-20M/MLP-20M/100003.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20260201%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260201T125134Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=d67ce01cbbc8538b80ce98f25d24b73c108aa0779a8a7ad251029cd434120a3cbedba9937d7d18546ffdd972a73e2f7dcf42980d1e79162209dd818de48ef283904f403e78e310e145f66ba1854add00ab2b9e21feea7877a40592371f77045fa39a82c9e246c303296abfacabd7a57933a82c2fdcb3f635e0c2da0264c7bd3127afcc9ed8489c6867bff088b64bc2bba5aa0e0b31ea2bf236a1b670f7d786673b5b34e154bf3fee0b0049d7e9f80950b6ba5b1deaa53f0b84549c26b9a89b7b32a7b0b074db9cb4037f411806992c473e881b77c75131cecd89495e4bace5a9e4065a535958ecb466f4e089c11b65aaf5fa105c5f852709290c9d428515e5a6"},
                    {"movieId": 200, "title": "The Matrix", "score": 4.79, "poster": "https://storage.googleapis.com/kagglesdsdata/datasets/38615/58801/MLP-20M/MLP-20M/100006.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20260201%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260201T125145Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=0b71906d763b8ca34bb0f951f32525f1a5c64e6b56412a940c232c949420acd31fb89d2bc2aa1c48bbaaddc184c670f8eba0650ebc183b9cfbbce8cb82c6e125f543ce3197b475e48d9d9ac6518b15ad06054a1b4353357b98c6133bb280f096a38dfa74dc7117b52e561a20e0631d17e1c61fc84999e30e84d2477b69aa96fe0ba26dd484e53d6c0eb81d7a447603b1322e22be26cabdbea005186c9eaebd09c40d6233e97b4e35ff6e6cf0d1a1632523b8aab4e4e3010d1d6ed1db3b2c466a4f70389963585e0bfe697ef77bde11d1a6f97a0579b7d2acd59708d3b03ac72c1ffda3b13856f089c8fc876480b8679d424c0012b2f3833d69c70293de035559"},
                ]
            }

            if response.status_code == 200:
                print(response.json())
                token = response.json()["access_token"]
                userid = response.json()["userid"]
                
                st.session_state["token"] = token
                st.session_state["username"] = username
                st.session_state["userid"] = userid
                st.session_state["recommendations"] = api_response["recommendations"]
                st.session_state.index = 0
                st.success("Connexion réussie !")
                st.rerun()
            else:
                st.error("Nom d'utilisateur ou mot de passe incorrect.")
            