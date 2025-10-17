"""
Script pour charger les donnees CSV dans la base de donnees SQLite
"""
import sqlite3
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def ingest_movies(conn, data_path):
    """
    Ingere les donnees movies.csv
    """
    print("\n1. Ingestion de movies.csv...")
    movies = pd.read_csv(data_path / "movies.csv")
    
    print(f"   Nombre de films: {len(movies):,}")
    
    # Inserer par batch pour plus de performance
    movies.to_sql('movies', conn, if_exists='append', index=False)
    
    print("   OK - Movies inseres")
    return len(movies)


def ingest_links(conn, data_path):
    """
    Ingere les donnees links.csv
    """
    print("\n2. Ingestion de links.csv...")
    links = pd.read_csv(data_path / "links.csv")
    
    print(f"   Nombre de liens: {len(links):,}")
    
    links.to_sql('links', conn, if_exists='append', index=False)
    
    print("   OK - Links inseres")
    return len(links)


def ingest_ratings(cursor, conn, data_path):
    """
    Ingere les donnees ratings.csv (le plus gros fichier)
    """
    print("\n3. Ingestion de ratings.csv (cela peut prendre quelques minutes)...")
    
    # Charger par chunks pour economiser la memoire
    chunk_size = 100000
    ratings_file = data_path / "ratings.csv"
    
    # Compter le nombre total de lignes pour la barre de progression
    total_lines = sum(1 for _ in open(ratings_file, encoding='utf-8')) - 1  # -1 pour le header
    print(f"   Nombre total de ratings: {total_lines:,}")
    
    # Charger et inserer par chunks
    chunks = pd.read_csv(ratings_file, chunksize=chunk_size)
    
    total_inserted = 0
    with tqdm(total=total_lines, desc="   Progression", unit=" ratings") as pbar:
        for chunk in chunks:
            chunk.to_sql('ratings', conn, if_exists='append', index=False)
            total_inserted += len(chunk)
            pbar.update(len(chunk))
    
    print(f"   OK - {total_inserted:,} ratings inseres")
    return total_inserted


def ingest_tags(cursor, conn, data_path):
    """
    Ingere les donnees tags.csv
    """
    print("\n4. Ingestion de tags.csv...")
    
    # Charger par chunks aussi (tags peut etre gros)
    chunk_size = 50000
    tags_file = data_path / "tags.csv"
    
    total_lines = sum(1 for _ in open(tags_file, encoding='utf-8')) - 1
    print(f"   Nombre total de tags: {total_lines:,}")
    
    chunks = pd.read_csv(tags_file, chunksize=chunk_size)
    
    total_inserted = 0
    skipped = 0
    with tqdm(total=total_lines, desc="   Progression", unit=" tags") as pbar:
        for chunk in chunks:
            # Supprimer les lignes avec des tags vides ou NULL
            chunk_clean = chunk.dropna(subset=['tag'])
            chunk_clean = chunk_clean[chunk_clean['tag'].str.strip() != '']
            
            skipped += len(chunk) - len(chunk_clean)
            
            if len(chunk_clean) > 0:
                chunk_clean.to_sql('tags', conn, if_exists='append', index=False)
                total_inserted += len(chunk_clean)
            
            pbar.update(len(chunk))
    
    print(f"   OK - {total_inserted:,} tags inseres ({skipped:,} tags vides ignores)")
    return total_inserted


def verify_data(cursor):
    """
    Verifie que les donnees ont bien ete inserees
    """
    print("\n" + "=" * 60)
    print("VERIFICATION DES DONNEES")
    print("=" * 60)
    
    tables = ['movies', 'ratings', 'tags', 'links']
    
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"{table:15} : {count:,} lignes")


def main():
    """
    Fonction principale d'ingestion
    """
    # Chemins
    project_root = Path(__file__).parent.parent.parent
    db_path = project_root / "database" / "recofilm.db"
    data_path = project_root / "data" / "raw" / "ml-20m"
    
    # Verifier que la BDD existe
    if not db_path.exists():
        print("ERREUR: La base de donnees n'existe pas!")
        print("Executez d'abord: python database/init_db.py")
        return
    
    # Verifier que les donnees existent
    if not data_path.exists():
        print("ERREUR: Les donnees n'existent pas!")
        print("Executez d'abord: python src/data/download_data.py")
        return
    
    print(f"Base de donnees: {db_path}")
    print(f"Donnees source: {data_path}")
    
    # Connexion
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Ingestion des donnees
        movies_count = ingest_movies(conn, data_path)
        links_count = ingest_links(conn, data_path)
        ratings_count = ingest_ratings(cursor, conn, data_path)
        tags_count = ingest_tags(cursor, conn, data_path)
        
        # Valider les changements
        conn.commit()
        
        # Verification
        verify_data(cursor)
        
        # Taille finale de la BDD
        db_size = db_path.stat().st_size / (1024 * 1024)  # MB
        print("\n" + "=" * 60)
        print(f"INGESTION TERMINEE AVEC SUCCES")
        print(f"Taille de la base de donnees: {db_size:.2f} MB")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nERREUR lors de l'ingestion: {e}")
        conn.rollback()
    
    finally:
        conn.close()


if __name__ == "__main__":
    main()