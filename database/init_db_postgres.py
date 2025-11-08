"""
Script pour creer la structure de la base de donnees PostgreSQL
"""
from pathlib import Path
import sys

# Ajouter le chemin pour importer config
sys.path.append(str(Path(__file__).parent))
from config import get_connection


def drop_tables(conn):
    """
    Supprime toutes les tables (pour repartir de zero)
    """
    print("\nSuppression des tables existantes...")
    
    cursor = conn.cursor()
    
    # Desactiver les foreign keys temporairement
    cursor.execute("SET session_replication_role = 'replica';")
    
    tables = ['tags', 'links', 'ratings', 'movies']
    for table in tables:
        cursor.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
        print(f"   Table {table} supprimee")
    
    # Reactiver les foreign keys
    cursor.execute("SET session_replication_role = 'origin';")
    
    conn.commit()


def create_tables(conn):
    """
    Cree les tables necessaires
    """
    print("\nCreation des tables...")
    
    cursor = conn.cursor()
    
    # Table MOVIES
    print("Creation de la table movies...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS movies (
            movieId INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            genres TEXT
        )
    """)
    
    # Table RATINGS
    print("Creation de la table ratings...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ratings (
            id SERIAL PRIMARY KEY,
            userId INTEGER NOT NULL,
            movieId INTEGER NOT NULL,
            rating REAL NOT NULL,
            timestamp INTEGER,
            FOREIGN KEY (movieId) REFERENCES movies(movieId)
        )
    """)
    
    # Index sur userId et movieId pour accelerer les requetes
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_ratings_userId 
        ON ratings(userId)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_ratings_movieId 
        ON ratings(movieId)
    """)
    
    # Table TAGS
    print("Creation de la table tags...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tags (
            id SERIAL PRIMARY KEY,
            userId INTEGER NOT NULL,
            movieId INTEGER NOT NULL,
            tag TEXT NOT NULL,
            timestamp INTEGER,
            FOREIGN KEY (movieId) REFERENCES movies(movieId)
        )
    """)
    
    # Table LINKS
    print("Creation de la table links...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS links (
            movieId INTEGER PRIMARY KEY,
            imdbId TEXT,
            tmdbId TEXT,
            FOREIGN KEY (movieId) REFERENCES movies(movieId)
        )
    """)
    
    conn.commit()
    print("Tables creees avec succes!")


def verify_tables(conn):
    """
    Verifie que les tables ont bien ete creees
    """
    print("\nVerification des tables...")
    
    cursor = conn.cursor()
    cursor.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name
    """)
    
    tables = cursor.fetchall()
    
    print("Tables creees:")
    for table in tables:
        print(f"  - {table[0]}")


def main():
    """
    Fonction principale
    """
    print("=" * 60)
    print("CREATION DE LA BASE DE DONNEES POSTGRESQL")
    print("=" * 60)
    
    try:
        # Connexion
        conn = get_connection()
        print("Connexion PostgreSQL etablie")
        
        # Demander confirmation pour supprimer les tables
        response = input("\nVoulez-vous supprimer les tables existantes? (y/n): ")
        if response.lower() == 'y':
            drop_tables(conn)
        
        # Creer les tables
        create_tables(conn)
        
        # Verifier
        verify_tables(conn)
        
        print("\n" + "=" * 60)
        print("CREATION TERMINEE AVEC SUCCES")
        print("=" * 60)
        
        conn.close()
        
    except Exception as e:
        print(f"\nERREUR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()