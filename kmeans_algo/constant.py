import os
from pathlib import Path

# Chemins relatifs basés sur le répertoire du script
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent

PATH_OUTPUT = str(SCRIPT_DIR / "output")
PATH_DATASET = str(PROJECT_DIR / "dataset")
MODEL_CLUSTERING = "kmeans"