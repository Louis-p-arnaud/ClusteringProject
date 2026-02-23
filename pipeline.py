from sklearn.preprocessing import StandardScaler
from Descriptors.vit_descriptor import ViTDescriptor # Ton nouveau fichier
from sklearn.cluster import SpectralClustering
import os
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import io

from Algos.kmeans_algo.kmeans import KMeans, show_metric
from Descriptors.features import compute_hog_descriptors, compute_color_histograms
from utils import *
from constant import PATH_OUTPUT, MODEL_CLUSTERING, PATH_DATASET


def load_images_from_dataset(dataset_path):
    """
    Charge les images depuis le dossier dataset SANS utiliser les noms des dossiers (clustering non supervisé).
    Input : dataset_path (str) : chemin vers le dossier dataset
    Output : images (list), labels_true (np.array optional pour évaluation)
    """
    images = []
    labels_true = []  # Gardé pour évaluation POST-clustering uniquement
    category_names = []
    image_paths = []
    
    # Parcourir les dossiers du dataset
    dataset_dir = Path(dataset_path)
    label = 0
    
    for category_folder in sorted(dataset_dir.iterdir()):
        if category_folder.is_dir():
            category_name = category_folder.name
            category_names.append(category_name)
            
            # Charger toutes les images du dossier
            for img_file in sorted(category_folder.glob("*")):
                if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                    try:
                        # Lire le fichier en bytes et décoder avec OpenCV
                        with open(img_file, 'rb') as f:
                            img_bytes = np.frombuffer(f.read(), np.uint8)
                        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
                        
                        if img is not None:
                            # Redimensionner à 64x64 pour cohérence
                            img = cv2.resize(img, (64, 64))
                            images.append(img)
                            image_paths.append(str(img_file))
                            # Labels gardés seulement pour évaluation POST clustering
                            labels_true.append(label)
                    except Exception as e:
                        print(f"Erreur lors du chargement de {img_file}: {e}")
            
            label += 1
    
    return np.array(images), np.array(labels_true), category_names, image_paths


def pipeline():
    print("\n\n ##### Chargement du dataset ######")
    images, labels_true, category_names, image_paths = load_images_from_dataset(PATH_DATASET)
    
    print("\n\n ##### Extraction de Features ######")
    print("- calcul features HOG...")
    descriptors_hog = compute_hog_descriptors(images)
    print("- calcul features Histogram...")
    descriptors_hist = compute_color_histograms(images)
    
    # --- AJOUT DU DESCRIPTEUR ViT ---
    print("- calcul features Vision Transformer (ViT)...")
    vit_extractor = ViTDescriptor()
    descriptors_vit = []
    for path in image_paths:
        feat = vit_extractor.get_features(path)
        descriptors_vit.append(feat)
    descriptors_vit = np.array(descriptors_vit)

    print("\n\n ##### Clustering (NON SUPERVISÉ) ######")
    number_cluster = 20 # Objectif fixé par le sujet [cite: 6]
    
    # Modèles existants
    kmeans_hog = KMeans(n_clusters=number_cluster, random_state=42)
    kmeans_hist = KMeans(n_clusters=number_cluster, random_state=42)
    
    # --- AJOUT DU SPECTRAL CLUSTERING ---
    print("- calcul Spectral Clustering avec features ViT...")
    # On utilise 'nearest_neighbors' pour la connectivité du graphe
    spectral_model = SpectralClustering(n_clusters=number_cluster, affinity='nearest_neighbors', assign_labels='kmeans', random_state=42)
    spectral_labels = spectral_model.fit_predict(descriptors_vit)

    print("- calcul kmeans classiques...")
    kmeans_hog.fit(np.array(descriptors_hog))
    kmeans_hist.fit(np.array(descriptors_hist))

    print("\n\n ##### Résultat ######")
    metric_hist = show_metric(labels_true, kmeans_hist.labels_, descriptors_hist, bool_show=True, name_descriptor="HISTOGRAM", bool_return=True)
    metric_hog = show_metric(labels_true, kmeans_hog.labels_, descriptors_hog, bool_show=True, name_descriptor="HOG", bool_return=True)
    
    # Métrique pour ton nouveau modèle
    metric_vit = show_metric(labels_true, spectral_labels, descriptors_vit, bool_show=True, name_descriptor="ViT_SPECTRAL", bool_return=True)

    print("- export des données vers le dashboard")
    df_metric = pd.DataFrame([metric_hist, metric_hog, metric_vit])
    
    # Normalisation et conversion 3D pour la visualisation 
    scaler = StandardScaler()
    x_3d_hist = conversion_3d(scaler.fit_transform(descriptors_hist))
    x_3d_hog = conversion_3d(scaler.fit_transform(descriptors_hog))
    x_3d_vit = conversion_3d(scaler.fit_transform(descriptors_vit))

    # Création des DataFrames d'export
    df_hist = create_df_to_export(x_3d_hist, labels_true, kmeans_hist.labels_, image_paths)
    df_hog = create_df_to_export(x_3d_hog, labels_true, kmeans_hog.labels_, image_paths)
    df_vit = create_df_to_export(x_3d_vit, labels_true, spectral_labels, image_paths)

    if not os.path.exists(PATH_OUTPUT):
        os.makedirs(PATH_OUTPUT)

    # Sauvegarde des fichiers Excel pour le Dashboard
    df_hist.to_excel(PATH_OUTPUT+"/save_clustering_hist_kmeans.xlsx")
    df_hog.to_excel(PATH_OUTPUT+"/save_clustering_hog_kmeans.xlsx")
    df_vit.to_excel(PATH_OUTPUT+"/save_clustering_vit_spectral.xlsx")
    df_metric.to_excel(PATH_OUTPUT+"/save_metric.xlsx")
    
    print("Fin. Pipeline exécutée avec succès.")

if __name__ == "__main__":
    pipeline()