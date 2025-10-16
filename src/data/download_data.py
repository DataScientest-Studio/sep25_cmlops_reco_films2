"""
Script to download and extract MovieLens 20M dataset
"""
import os
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm


def download_file(url, destination):
    """
    Download a file with progress bar
    
    Args:
        url (str): URL to download from
        destination (str): Path where to save the file
    """
    print(f"Downloading from {url}...")
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as file, tqdm(
        desc=destination,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)
    
    print(f" Download completed: {destination}")


def extract_zip(zip_path, extract_to):
    """
    Extract a zip file
    
    Args:
        zip_path (str): Path to the zip file
        extract_to (str): Directory where to extract
    """
    print(f"Extracting {zip_path}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    print(f" Extraction completed to: {extract_to}")


def main():
    """
    Main function to download and extract MovieLens 20M
    """
    # Define paths
    project_root = Path(__file__).parent.parent.parent
    data_raw_path = project_root / "data" / "raw"
    
    # Create directories if they don't exist
    data_raw_path.mkdir(parents=True, exist_ok=True)
    
    # MovieLens 20M URL
    url = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"
    zip_filename = "ml-20m.zip"
    zip_path = data_raw_path / zip_filename
    
    # Check if data already exists
    if (data_raw_path / "ml-20m").exists():
        print("  MovieLens 20M data already exists in data/raw/ml-20m/")
        response = input("Do you want to re-download? (y/n): ")
        if response.lower() != 'y':
            print("Skipping download.")
            return
    
    # Download the file
    download_file(url, str(zip_path))
    
    # Extract the file
    extract_zip(str(zip_path), str(data_raw_path))
    
    # Remove the zip file to save space
    print("Cleaning up zip file...")
    os.remove(zip_path)
    print(" Cleanup completed")
    
    # Verify files
    ml_20m_path = data_raw_path / "ml-20m"
    expected_files = [
        "movies.csv",
        "ratings.csv",
        "tags.csv",
        "links.csv",
        "genome-scores.csv",
        "genome-tags.csv"
    ]
    
    print("\n Verifying downloaded files:")
    all_files_exist = True
    for filename in expected_files:
        file_path = ml_20m_path / filename
        if file_path.exists():
            file_size = file_path.stat().st_size / (1024 * 1024)  # Size in MB
            print(f"   {filename} ({file_size:.2f} MB)")
        else:
            print(f"   {filename} - NOT FOUND")
            all_files_exist = False
    
    if all_files_exist:
        print("\n All files downloaded successfully!")
        print(f" Data location: {ml_20m_path}")
    else:
        print("\n  Some files are missing. Please check the download.")


if __name__ == "__main__":
    main()