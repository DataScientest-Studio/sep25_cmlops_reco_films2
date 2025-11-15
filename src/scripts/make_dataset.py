#!/usr/bin/env python3
import os
import shutil
import requests
import zipfile
from pathlib import Path
from typing import List

def make_dataset(url: str, repo_data: str, overwrite: bool = False) -> List[Path]:
    """
    Télécharge et extrait le dataset contenu de l'url puis
    déplace les fichiers du dossier extrait (ex: 'ml-20m') vers repo_data.

    Args:
        url (str): URL directe du fichier ZIP.
        repo_data (str): Chemin du répertoire où enregistrer et extraire les fichiers.
        overwrite (bool): Si True, les fichiers existants dans repo_data seront écrasés.
                           Si False, les fichiers existants seront préservés (les nouveaux seront sautés).

    Returns:
        List[Path]: liste des chemins déplacés dans repo_data.
    """
    repo_path = Path(repo_data)
    repo_path.mkdir(parents=True, exist_ok=True)

    chemin_zip = repo_path / "ml-20m.zip"

    # Téléchargement du fichier
    print(f"Téléchargement du dataset depuis {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(chemin_zip, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"Dataset téléchargé : {chemin_zip}")

    # Extraction du fichier ZIP
    print(f"Extraction du fichier ZIP dans {repo_path}...")
    with zipfile.ZipFile(chemin_zip, 'r') as zip_ref:
        zip_ref.extractall(repo_path)
    print(f"Fichiers extraits dans {repo_path}")

    # Supprimer le ZIP
    try:
        chemin_zip.unlink()
        print(f"Fichier ZIP supprimé : {chemin_zip}")
    except Exception as e:
        print(f"Impossible de supprimer {chemin_zip}: {e}")

    # Trouver le dossier extrait contenant le dataset (ex: 'ml-20m')
    extracted_dir = None
    candidate = repo_path / "ml-20m"
    if candidate.exists() and candidate.is_dir():
        extracted_dir = candidate
    else:
        # Recherche plus souple : dossier qui contient movies.csv ou ratings.csv
        for child in repo_path.iterdir():
            if child.is_dir():
                if any((child / fname).exists() for fname in ("movies.csv", "ratings.csv", "tags.csv")):
                    extracted_dir = child
                    break

    if extracted_dir is None:
        print("Aucun dossier extrait 'ml-20m' détecté. Rien à déplacer.")
        return []

    print(f"Dossier extrait détecté : {extracted_dir}")
    moved_paths = []

    # Déplacer chaque élément du dossier extrait vers le dossier parent (repo_path)
    for item in extracted_dir.iterdir():
        target = repo_path / item.name

        # Si cible existe
        if target.exists():
            if overwrite:
                # supprimer cible avant déplacement
                try:
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                    print(f"Remplacé: {target}")
                except Exception as e:
                    print(f"Erreur lors de la suppression de {target}: {e}")
                    continue
            else:
                print(f"Existant - skip: {target}")
                continue

        try:
            shutil.move(str(item), str(repo_path))
            moved_paths.append(target)
            print(f"Déplacé: {target}")
        except Exception as e:
            print(f"Erreur lors du déplacement {item} -> {target}: {e}")

    # Tenter de supprimer le dossier extrait s'il est vide
    try:
        # si dossier toujours présent et vide -> remove, sinon essayer de rmdir (échec si non vide)
        if extracted_dir.exists():
            try:
                extracted_dir.rmdir()
                print(f"Dossier extrait supprimé : {extracted_dir}")
            except OSError:
                # s'il reste des choses, on tente un rmtree uniquement si overwrite True (dangerosité)
                if overwrite:
                    shutil.rmtree(extracted_dir)
                    print(f"Dossier extrait supprimé récursivement : {extracted_dir}")
                else:
                    print(f"Le dossier {extracted_dir} n'est pas vide ; il reste des fichiers non déplacés.")
    except Exception as e:
        print(f"Impossible de supprimer le dossier extrait {extracted_dir}: {e}")

    return moved_paths


if __name__ == "__main__":
    url = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"
    raw_data = "./data/raw_data"

    moved = make_dataset(url, raw_data, overwrite=True)
    print(f"Fichiers déplacés ({len(moved)}):")
    for p in moved:
        print(" -", p)
