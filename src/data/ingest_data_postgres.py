"""
Script pour ingerer les donnees CSV vers PostgreSQL (Supabase)
Adapte pour respecter la limite de 500 MB de Supabase gratuit
"""
import pandas as pd
from pathlib import Path
import sys
import time

# Ajouter le chemin pour importer config
sys.path.append(str(Path(__file__).resolve().parents[2] / "database"))
from config import get_connection
from psycopg2.extras import execute_batch


def estimate_data_size(data_dir):
    """
    Estime la taille des donnees a inserer
    """
    print("\n" + "=" * 60)
    print("ESTIMATION DE LA TAILLE DES DONNEES")
    print("=" * 60)
    
    files = {
        'movies.csv': 0,
        'ratings.csv': 0,
        'tags.csv': 0,
        'links.csv': 0
    }
    
    for filename in files.keys():
        filepath = data_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            files[filename] = size_mb
            print(f"  {filename}: {size_mb:.2f} MB")
    
    total = sum(files.values())
    print(f"\nTotal: {total:.2f} MB")
    
    if total > 400:  # Marge de securite pour 500 MB
        print("\n⚠️  WARNING: Les donnees depassent 400 MB!")
        print("💡 Solution: Echantillonnage des ratings (fichier le plus gros)")
        return True, total
    
    return False, total


def insert_movies(conn, data_dir):
    """
    Insere les donnees de la table movies
    """
    print("\n" + "=" * 60)
    print("INSERTION DES FILMS")
    print("=" * 60)
    
    movies_file = data_dir / "movies.csv"
    
    if not movies_file.exists():
        print(f"❌ Erreur: {movies_file} n'existe pas!")
        return
    
    df = pd.read_csv(movies_file)
    print(f"Nombre de films a inserer: {len(df):,}")
    
    cursor = conn.cursor()
    
    # Preparer les donnees
    data = [(int(row['movieId']), row['title'], row['genres']) 
            for _, row in df.iterrows()]
    
    # Insertion par batch pour performance
    print("Insertion en cours...")
    execute_batch(cursor, """
        INSERT INTO movies (movieId, title, genres)
        VALUES (%s, %s, %s)
        ON CONFLICT (movieId) DO NOTHING
    """, data, page_size=1000)
    
    conn.commit()
    cursor.close()
    
    print(f"✅ {len(df):,} films inseres avec succes!")


def insert_ratings(conn, data_dir, sample_size=None):
    """
    Insere les donnees de la table ratings
    
    Args:
        sample_size: Si defini, limite le nombre de ratings (pour respecter 500 MB)
    """
    print("\n" + "=" * 60)
    print("INSERTION DES RATINGS")
    print("=" * 60)
    
    ratings_file = data_dir / "ratings.csv"
    
    if not ratings_file.exists():
        print(f"❌ Erreur: {ratings_file} n'existe pas!")
        return
    
    if sample_size:
        print(f"⚠️  Mode echantillonne: {sample_size:,} ratings sur ~20M")
        df = pd.read_csv(ratings_file, nrows=sample_size)
    else:
        print("Chargement de tous les ratings...")
        df = pd.read_csv(ratings_file)
    
    print(f"Nombre de ratings a inserer: {len(df):,}")
    
    cursor = conn.cursor()
    
    # Insertion par chunks pour eviter les timeouts
    chunk_size = 10000
    total_chunks = len(df) // chunk_size + 1
    
    print("\nInsertion en cours (par chunks de 10,000)...")
    start_time = time.time()
    
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        chunk_num = i // chunk_size + 1
        
        data = [(int(row['userId']), int(row['movieId']), 
                 float(row['rating']), int(row['timestamp'])) 
                for _, row in chunk.iterrows()]
        
        execute_batch(cursor, """
            INSERT INTO ratings (userId, movieId, rating, timestamp)
            VALUES (%s, %s, %s, %s)
        """, data, page_size=1000)
        
        conn.commit()
        
        # Afficher la progression toutes les 10 chunks
        if chunk_num % 10 == 0 or chunk_num == total_chunks:
            elapsed = time.time() - start_time
            progress = (i + len(chunk)) / len(df) * 100
            print(f"  Progression: {progress:.1f}% ({i+len(chunk):,}/{len(df):,}) - {elapsed:.1f}s")
    
    cursor.close()
    
    total_time = time.time() - start_time
    print(f"\n✅ {len(df):,} ratings inseres en {total_time:.1f} secondes!")


def insert_tags(conn, data_dir):
    """
    Insere les donnees de la table tags
    """
    print("\n" + "=" * 60)
    print("INSERTION DES TAGS")
    print("=" * 60)
    
    tags_file = data_dir / "tags.csv"
    
    if not tags_file.exists():
        print(f"❌ Erreur: {tags_file} n'existe pas!")
        return
    
    df = pd.read_csv(tags_file)
    print(f"Nombre de tags a inserer: {len(df):,}")
    
    cursor = conn.cursor()
    
    data = [(int(row['userId']), int(row['movieId']), 
             row['tag'], int(row['timestamp'])) 
            for _, row in df.iterrows()]
    
    print("Insertion en cours...")
    execute_batch(cursor, """
        INSERT INTO tags (userId, movieId, tag, timestamp)
        VALUES (%s, %s, %s, %s)
    """, data, page_size=1000)
    
    conn.commit()
    cursor.close()
    
    print(f"✅ {len(df):,} tags inseres avec succes!")


def insert_links(conn, data_dir):
    """
    Insere les donnees de la table links
    """
    print("\n" + "=" * 60)
    print("INSERTION DES LINKS")
    print("=" * 60)
    
    links_file = data_dir / "links.csv"
    
    if not links_file.exists():
        print(f"❌ Erreur: {links_file} n'existe pas!")
        return
    
    df = pd.read_csv(links_file)
    print(f"Nombre de liens a inserer: {len(df):,}")
    
    cursor = conn.cursor()
    
    data = [(int(row['movieId']), str(row['imdbId']), str(row['tmdbId'])) 
            for _, row in df.iterrows()]
    
    print("Insertion en cours...")
    execute_batch(cursor, """
        INSERT INTO links (movieId, imdbId, tmdbId)
        VALUES (%s, %s, %s)
        ON CONFLICT (movieId) DO NOTHING
    """, data, page_size=1000)
    
    conn.commit()
    cursor.close()
    
    print(f"✅ {len(df):,} liens inseres avec succes!")


def verify_data(conn):
    """
    Verifie que les donnees ont bien ete inserees
    """
    print("\n" + "=" * 60)
    print("VERIFICATION DES DONNEES INSEREES")
    print("=" * 60)
    
    cursor = conn.cursor()
    
    tables = ['movies', 'ratings', 'tags', 'links']
    
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table};")
        count = cursor.fetchone()[0]
        print(f"  {table}: {count:,} lignes")
    
    cursor.close()


def main():
    """
    Fonction principale pour ingerer toutes les donnees
    """
    print("\n" + "=" * 60)
    print("INGESTION DES DONNEES VERS POSTGRESQL")
    print("=" * 60)
    
    # Chemin vers les donnees
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data" / "raw" / "ml-20m"
    
    print(f"\nChemin des donnees: {data_dir}")
    
    if not data_dir.exists():
        print(f"\n❌ Erreur: Le dossier {data_dir} n'existe pas!")
        print("💡 Assurez-vous d'avoir telecharge et extrait les donnees MovieLens")
        return
    
    try:
        # Estimer la taille des donnees
        need_sampling, total_size = estimate_data_size(data_dir)
        
        # Demander confirmation
        print("\n" + "=" * 60)
        if need_sampling:
            print("⚠️  Les donnees depassent 400 MB")
            print("💡 Recommandation: Echantillonner a 10M ratings")
            response = input("\nVoulez-vous echantillonner les ratings? (y/n): ")
            sample_ratings = response.lower() == 'y'
        else:
            response = input("\nCommencer l'ingestion? (y/n): ")
            if response.lower() != 'y':
                print("Ingestion annulee")
                return
            sample_ratings = False
        
        # Connexion a la base
        print("\nConnexion a PostgreSQL...")
        conn = get_connection()
        print("✅ Connexion etablie")
        
        start_time = time.time()
        
        # Insertion des donnees dans l'ordre (movies en premier pour les foreign keys)
        insert_movies(conn, data_dir)
        
        if sample_ratings and need_sampling:
            insert_ratings(conn, data_dir, sample_size=10000000)  # 10M ratings
        else:
            insert_ratings(conn, data_dir)
        
        insert_tags(conn, data_dir)
        insert_links(conn, data_dir)
        
        # Verification finale
        verify_data(conn)
        
        conn.close()
        
        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        
        print("\n" + "=" * 60)
        print(f"✅ INGESTION TERMINEE EN {minutes}min {seconds}s!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 60)


if __name__ == "__main__":
    main()
