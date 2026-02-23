# Image de base Python
FROM python:3.9-slim

# Répertoire de travail
WORKDIR /app

# Installation des dépendances
# Attention : j'utilise votre orthographe "requierements.txt"
COPY requierements.txt .
RUN pip install --no-cache-dir -r requierements.txt

# Copie du projet (scripts, dataset, dossiers)
COPY . .

# Port pour le Dashboard Streamlit
EXPOSE 8501

# Par défaut, lance le pipeline IA au build/run
CMD ["python", "pipeline.py"]