import os
from pathlib import Path

# Chemins relatifs basés sur le répertoire du script
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = os.path.abspath(SCRIPT_DIR)  # Assure que nous sommes dans le répertoire du projet


PATH_ALGO = os.path.join(PROJECT_DIR, "Algos")
PATH_OUTPUT = os.path.join(PATH_ALGO, "kmeans_algo", "output")
PATH_DATASET = os.environ.get("DATA_PATH", os.path.join(PROJECT_DIR, "data", "test"))
MODEL_CLUSTERING = "kmeans"