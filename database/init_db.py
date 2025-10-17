"""
Script pour creer la structure de la base de donnees SQLite
"""
import sqlite3
from pathlib import Path


def create_database():
    """
    Cree la base de donnees et les tables necessaires
    """
    # Chemin de la base de donnees
    project_root = Path(__file__).parent.parent
    db_path = project_root / "database" / "recofilm.db"
    
    # Creer le dossier database s'il n'existe pas
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Supprimer la BDD si elle existe deja (pour repartir de zero)
    if db_path.exists():
        print(f"Suppression de l'ancienne base de donnees: {db_path}")
        db_path.unlink()
    
    # Connexion a la base de donnees (cree le fichier)
    print(f"Creation de la base de donnees: {db_path}")
    conn = sqlite3.connect(db_path)
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
            id INTEGER PRIMARY KEY AUTOINCREMENT,
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
            id INTEGER PRIMARY KEY AUTOINCREMENT,
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
    
    # Valider les changements
    conn.commit()
    
    # Verifier les tables creees
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table'
    """)
    tables = cursor.fetchall()
    
    print("\nTables creees avec succes:")
    for table in tables:
        print(f"  - {table[0]}")
    
    # Fermer la connexion
    conn.close()
    
    print(f"\nBase de donnees creee: {db_path}")
    print(f"Taille: {db_path.stat().st_size / 1024:.2f} KB")
    
    return db_path


if __name__ == "__main__":
    create_database()