"""
Script d'entrainement du modele de recommandation (KNN)
"""
import pandas as pd
import pickle
from pathlib import Path
from sklearn.neighbors import NearestNeighbors


def train_model(movie_matrix_path):
    """
    Entraine un modele KNN sur la movie_matrix
    
    Args:
        movie_matrix_path: Chemin vers movie_matrix.csv
    
    Returns:
        model: Modele KNN entraine
        movie_ids: Liste des movieIds correspondant aux indices du modele
    """
    print("\n1. Chargement de movie_matrix...")
    movie_matrix = pd.read_csv(movie_matrix_path)
    print(f"   Nombre de films: {len(movie_matrix):,}")
    print(f"   Nombre de features: {movie_matrix.shape[1] - 1}")
    
    # Separer les movieIds des features
    movie_ids = movie_matrix['movieId'].values
    features = movie_matrix.drop('movieId', axis=1)
    
    print(f"\n2. Entrainement du modele KNN...")
    print(f"   Algorithme: ball_tree")
    print(f"   Nombre de voisins: 20")
    
    # Entrainer le modele KNN
    model = NearestNeighbors(n_neighbors=20, algorithm='ball_tree')
    model.fit(features)
    
    print(f"   Modele entraine avec succes!")
    
    return model, movie_ids


def save_model(model, movie_ids, output_dir):
    """
    Sauvegarde le modele et les movieIds
    
    Args:
        model: Modele KNN entraine
        movie_ids: Liste des movieIds
        output_dir: Dossier de sortie
    """
    print(f"\n3. Sauvegarde du modele...")
    
    # Creer le dossier s'il n'existe pas
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sauvegarder le modele
    model_path = output_dir / "model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"   Modele sauvegarde: {model_path}")
    print(f"   Taille: {model_path.stat().st_size / 1024:.2f} KB")
    
    # Sauvegarder les movieIds (pour retrouver les films plus tard)
    ids_path = output_dir / "movie_ids.pkl"
    with open(ids_path, 'wb') as f:
        pickle.dump(movie_ids, f)
    
    print(f"   MovieIds sauvegardes: {ids_path}")


def test_model(model, movie_ids, movie_matrix_path):
    """
    Test rapide du modele avec un film exemple
    
    Args:
        model: Modele KNN entraine
        movie_ids: Liste des movieIds
        movie_matrix_path: Chemin vers movie_matrix.csv
    """
    print(f"\n4. Test du modele...")
    
    # Charger movie_matrix pour avoir les titres
    movie_matrix = pd.read_csv(movie_matrix_path)
    
    # Tester avec le premier film
    test_film_idx = 0
    test_film_id = movie_ids[test_film_idx]
    
    # Trouver les films similaires
    features = movie_matrix.drop('movieId', axis=1)
    distances, indices = model.kneighbors([features.iloc[test_film_idx]])
    
    # Afficher les resultats
    print(f"\n   Film de test: movieId={test_film_id}")
    print(f"   Films similaires trouves:")
    
    for i, (dist, idx) in enumerate(zip(distances[0][1:6], indices[0][1:6])):
        similar_movie_id = movie_ids[idx]
        print(f"     {i+1}. movieId={similar_movie_id} (distance={dist:.4f})")
    
    print(f"\n   Le modele fonctionne correctement!")


def main():
    """
    Fonction principale d'entrainement
    """
    print("=" * 60)
    print("ENTRAINEMENT DU MODELE DE RECOMMANDATION")
    print("=" * 60)
    
    # Chemins
    project_root = Path(__file__).parent.parent.parent
    movie_matrix_path = project_root / "data" / "processed" / "movie_matrix.csv"
    output_dir = project_root / "models"
    
    # Verifier que movie_matrix existe
    if not movie_matrix_path.exists():
        print(f"\nERREUR: {movie_matrix_path} n'existe pas!")
        print("Executez d'abord: python src/data/preprocess.py")
        return
    
    try:
        # Entrainer le modele
        model, movie_ids = train_model(movie_matrix_path)
        
        # Sauvegarder
        save_model(model, movie_ids, output_dir)
        
        # Tester
        test_model(model, movie_ids, movie_matrix_path)
        
        print("\n" + "=" * 60)
        print("ENTRAINEMENT TERMINE AVEC SUCCES")
        print("=" * 60)
        print(f"\nLe modele est pret a etre utilise!")
        print(f"Prochaine etape: python src/models/predict_model.py")
        
    except Exception as e:
        print(f"\nERREUR lors de l'entrainement: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()