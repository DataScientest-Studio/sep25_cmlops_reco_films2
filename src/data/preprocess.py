"""
Script de preprocessing pour creer les matrices movie_matrix et user_matrix
"""
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm


def create_movie_matrix(conn):
    """
    Cree la movie_matrix : chaque ligne = un film avec ses features
    """
    print("\n1. Creation de movie_matrix...")
    
    # Charger les films avec leurs genres
    movies = pd.read_sql_query("SELECT movieId, genres FROM movies", conn)
    print(f"   Nombre de films: {len(movies):,}")
    
    # Extraire tous les genres uniques
    all_genres = set()
    for genres_str in movies['genres']:
        if genres_str and genres_str != '(no genres listed)':
            genres = genres_str.split('|')
            all_genres.update(genres)
    
    genres_list = sorted(list(all_genres))
    print(f"   Nombre de genres: {len(genres_list)}")
    print(f"   Genres: {', '.join(genres_list)}")
    
    # One-hot encoding des genres
    print("   One-hot encoding des genres...")
    for genre in genres_list:
        movies[genre] = movies['genres'].apply(
            lambda x: 1 if x and genre in x.split('|') else 0
        )
    
    # Calculer les statistiques des films (note moyenne, nb de ratings)
    print("   Calcul des statistiques des films...")
    stats = pd.read_sql_query("""
        SELECT 
            movieId,
            AVG(rating) as avg_rating,
            COUNT(*) as num_ratings
        FROM ratings
        GROUP BY movieId
    """, conn)
    
    # Fusionner avec les films
    movie_matrix = movies.merge(stats, on='movieId', how='left')
    
    # Remplir les NaN (films sans ratings)
    movie_matrix['avg_rating'].fillna(0, inplace=True)
    movie_matrix['num_ratings'].fillna(0, inplace=True)
    
    # Supprimer la colonne genres (on a les colonnes one-hot maintenant)
    movie_matrix.drop('genres', axis=1, inplace=True)
    
    print(f"   Dimensions de movie_matrix: {movie_matrix.shape}")
    print(f"   Colonnes: {list(movie_matrix.columns)}")
    
    return movie_matrix, genres_list


def create_user_matrix(conn, genres_list, rating_threshold=4.0):
    """
    Cree la user_matrix : chaque ligne = un utilisateur avec ses preferences
    """
    print(f"\n2. Creation de user_matrix (seuil de rating >= {rating_threshold})...")
    
    # Recuperer tous les utilisateurs
    users = pd.read_sql_query("SELECT DISTINCT userId FROM ratings", conn)
    print(f"   Nombre d'utilisateurs: {len(users):,}")
    
    # Initialiser la matrice utilisateur
    user_matrix = pd.DataFrame()
    user_matrix['userId'] = users['userId']
    
    # Initialiser les colonnes de genres
    for genre in genres_list:
        user_matrix[genre] = 0.0
    
    user_matrix['avg_rating_given'] = 0.0
    user_matrix['num_ratings_given'] = 0
    
    print("   Calcul des preferences utilisateurs...")
    
    # Pour chaque utilisateur, calculer ses preferences
    # On va traiter par batch pour economiser la memoire
    batch_size = 1000
    num_batches = len(users) // batch_size + 1
    
    for batch_idx in tqdm(range(num_batches), desc="   Progression"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(users))
        batch_users = users.iloc[start_idx:end_idx]['userId'].tolist()
        
        # Requete pour ce batch d'utilisateurs
        query = f"""
            SELECT 
                r.userId,
                m.genres,
                r.rating
            FROM ratings r
            JOIN movies m ON r.movieId = m.movieId
            WHERE r.userId IN ({','.join(map(str, batch_users))})
            AND r.rating >= {rating_threshold}
        """
        
        batch_data = pd.read_sql_query(query, conn)
        
        # Pour chaque utilisateur du batch
        for user_id in batch_users:
            user_ratings = batch_data[batch_data['userId'] == user_id]
            
            if len(user_ratings) == 0:
                continue
            
            # Calculer les preferences de genres (moyenne)
            genre_counts = {genre: 0 for genre in genres_list}
            total_movies = 0
            
            for genres_str in user_ratings['genres']:
                if genres_str and genres_str != '(no genres listed)':
                    genres = genres_str.split('|')
                    for genre in genres:
                        if genre in genre_counts:
                            genre_counts[genre] += 1
                    total_movies += 1
            
            # Normaliser (pourcentage)
            if total_movies > 0:
                for genre in genres_list:
                    user_matrix.loc[user_matrix['userId'] == user_id, genre] = \
                        genre_counts[genre] / total_movies
            
            # Statistiques utilisateur
            user_matrix.loc[user_matrix['userId'] == user_id, 'avg_rating_given'] = \
                user_ratings['rating'].mean()
            user_matrix.loc[user_matrix['userId'] == user_id, 'num_ratings_given'] = \
                len(user_ratings)
    
    print(f"   Dimensions de user_matrix: {user_matrix.shape}")
    print(f"   Colonnes: {list(user_matrix.columns)}")
    
    # Statistiques sur les utilisateurs
    active_users = user_matrix[user_matrix['num_ratings_given'] > 0]
    print(f"   Utilisateurs actifs (avec ratings >= {rating_threshold}): {len(active_users):,}")
    print(f"   Moyenne de films aimes par utilisateur: {active_users['num_ratings_given'].mean():.2f}")
    
    return user_matrix


def save_matrices(movie_matrix, user_matrix, output_path):
    """
    Sauvegarde les matrices dans data/processed/
    """
    print("\n3. Sauvegarde des matrices...")
    
    # Creer le dossier s'il n'existe pas
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Sauvegarder movie_matrix
    movie_path = output_path / "movie_matrix.csv"
    movie_matrix.to_csv(movie_path, index=False)
    print(f"   movie_matrix sauvegardee: {movie_path}")
    print(f"   Taille: {movie_path.stat().st_size / (1024*1024):.2f} MB")
    
    # Sauvegarder user_matrix
    user_path = output_path / "user_matrix.csv"
    user_matrix.to_csv(user_path, index=False)
    print(f"   user_matrix sauvegardee: {user_path}")
    print(f"   Taille: {user_path.stat().st_size / (1024*1024):.2f} MB")


def main():
    """
    Fonction principale
    """
    print("=" * 60)
    print("PREPROCESSING - CREATION DES MATRICES")
    print("=" * 60)
    
    # Chemins
    project_root = Path(__file__).parent.parent.parent
    db_path = project_root / "database" / "recofilm.db"
    output_path = project_root / "data" / "processed"
    
    # Verifier que la BDD existe
    if not db_path.exists():
        print("ERREUR: La base de donnees n'existe pas!")
        print("Executez d'abord: python src/data/ingest_data.py")
        return
    
    print(f"Base de donnees: {db_path}")
    print(f"Dossier de sortie: {output_path}")
    
    # Connexion
    conn = sqlite3.connect(db_path)
    
    try:
        # Creer les matrices
        movie_matrix, genres_list = create_movie_matrix(conn)
        user_matrix = create_user_matrix(conn, genres_list)
        
        # Sauvegarder
        save_matrices(movie_matrix, user_matrix, output_path)
        
        print("\n" + "=" * 60)
        print("PREPROCESSING TERMINE AVEC SUCCES")
        print("=" * 60)
        print("\nProchaines etapes:")
        print("1. Entrainer le modele: python src/models/train_model.py")
        print("2. Faire des predictions: python src/models/predict_model.py")
        
    except Exception as e:
        print(f"\nERREUR lors du preprocessing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        conn.close()


if __name__ == "__main__":
    main()