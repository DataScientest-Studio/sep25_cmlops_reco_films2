"""
Configuration pour la connexion PostgreSQL
"""
import os
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_batch

load_dotenv()


def get_connection():
    """
    Retourne une connexion PostgreSQL
    """
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )


def test_connection():
    """
    Test la connexion PostgreSQL
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print(f"Connexion PostgreSQL reussie!")
        print(f"Version: {version[0]}")
        conn.close()
        return True
    except Exception as e:
        print(f"Erreur de connexion: {e}")
        return False


if __name__ == "__main__":
    test_connection()