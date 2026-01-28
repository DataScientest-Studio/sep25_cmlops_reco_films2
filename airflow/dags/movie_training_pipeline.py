from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import requests
import logging

# Configuration
TRAINER_API_URL = "http://movie_trainer_api:8000"

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'movie_training_pipeline',
    default_args=default_args,
    description='Pipeline de collecte et training',
    schedule_interval='@daily',
    catchup=False,
    tags=['ml', 'movies'],
)


def insert_data():
    """Insère un chunk de données (ratings, tags, genome_scores)"""
    logging.info("🔍 Insertion de données...")
    
    response = requests.post(
        f"{TRAINER_API_URL}/insert-data",
        json={"force_insert": False},  # force_insert à False par défaut
        timeout=60
    )
    response.raise_for_status()
    
    data = response.json()
    
    if data.get("status") == "no_insertion_needed":
        logging.info(f"⏭️  Aucune insertion nécessaire : {data.get('message')}")
    else:
        logging.info(f"✅ Données insérées : {data}")
        logging.info(f"   - Ratings: {data['results']['ratings']['inserted_rows']} lignes")
        logging.info(f"   - Tags: {data['results']['tags']['inserted_rows']} lignes")
        logging.info(f"   - Genome-scores: {data['results']['genome-scores']['inserted_rows']} lignes")
    
    return data


def trigger_training():
    """Déclenche le training du modèle"""
    logging.info("🚀 Déclenchement du training...")
    
    response = requests.post(
        f"{TRAINER_API_URL}/training",
        json={
            "model_type": "svd",
            "params": {
                "n_factors": 100,
                "n_epochs": 20
            }
        },
        timeout=600
    )
    response.raise_for_status()
    
    result = response.json()
    logging.info(f"✅ Training terminé : {result}")
    return result


# Définition des tâches
task_insert = PythonOperator(
    task_id='insert_data',
    python_callable=insert_data,
    dag=dag,
)

task_train = PythonOperator(
    task_id='trigger_training',
    python_callable=trigger_training,
    dag=dag,
)

# Ordre d'exécution
task_insert >> task_train
