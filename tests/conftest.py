"""
Configuration pytest pour les tests de l'API
"""
import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Ajouter le chemin src au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.main import app


@pytest.fixture
def client():
    """
    Fixture qui fournit un client de test pour l'API
    """
    return TestClient(app)