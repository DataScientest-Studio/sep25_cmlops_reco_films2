"""
Tests unitaires pour l'API RecoFilm
"""
import pytest
import os


def test_read_root(client):
    """
    Test de la page d'accueil
    """
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "RecoFilm" in response.json()["message"]


def test_health_check(client):
    """
    Test du endpoint /health
    """
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    
    # Vérifie la structure de la réponse
    assert "status" in data
    assert "model_loaded" in data
    assert "database_connected" in data
    
    # Sur le CI, le modèle peut ne pas être chargé
    # On vérifie juste que le champ existe
    assert isinstance(data["model_loaded"], bool)


@pytest.mark.skipif(
    not os.path.exists("models/model.pkl"),
    reason="Modèle non disponible (normal sur CI)"
)
def test_predict_endpoint(client):
    """
    Test du endpoint /predict avec un utilisateur valide
    (Skip si le modèle n'existe pas)
    """
    payload = {
        "userId": 1,
        "numRecommendations": 5
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    
    # Vérifie la structure de la réponse
    assert "userId" in data
    assert "numRecommendations" in data
    assert "recommendations" in data
    
    # Vérifie que les recommandations sont présentes
    assert len(data["recommendations"]) <= 5
    
    # Vérifie la structure d'une recommandation
    if len(data["recommendations"]) > 0:
        rec = data["recommendations"][0]
        assert "movieId" in rec
        assert "title" in rec
        assert "genres" in rec


def test_predict_invalid_user(client):
    """
    Test du endpoint /predict avec un utilisateur invalide
    """
    payload = {
        "userId": 999999,  # Utilisateur qui n'existe pas
        "numRecommendations": 5
    }
    
    response = client.post("/predict", json=payload)
    # Devrait retourner une erreur 404 ou 500
    assert response.status_code in [404, 500]