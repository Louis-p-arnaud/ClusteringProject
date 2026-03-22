from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import io
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Algos.kmeans_algo.clustering import KMeans, show_metric
from Descriptors.ResNet50 import compute_resnet_descriptors, prepare_for_clustering
from Descriptors.features import compute_hog_descriptors, compute_color_histograms, compute_clip_descriptors
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
                            # Labels gardés seulement pour évaluation POST clustering
                            labels_true.append(label)
                            image_paths.append(str(img_file))
                    except Exception as e:
                        print(f"Erreur lors du chargement de {img_file}: {e}")
            
            label += 1
    
    return np.array(images), np.array(labels_true), category_names, image_paths


def pipeline():
   
    print("\n\n ##### Chargement du dataset ######")
    images, labels_true, category_names, image_paths = load_images_from_dataset(PATH_DATASET)
    
    print(f"- {len(images)} images chargées")
    print(f"- {len(category_names)} dossiers trouvés (pas utilisés pour le clustering non supervisé)")
   
    print("\n\n ##### Extraction de Features ######")
    print("- calcul features HOG...")
    descriptors_hog = compute_hog_descriptors(images)
    print("- calcul features Histogram...")
    descriptors_hist = compute_color_histograms(images)
    print("- calcul features ResNet50...")
    descriptors_resnet = compute_resnet_descriptors(images)
    #descriptors_resnet = prepare_for_clustering(descriptors_resnet)
    print("- calcul features CLIP...")
    descriptors_clip = compute_clip_descriptors(images)


    print("\n\n ##### Clustering (NON SUPERVISÉ) ######")
    number_cluster = 20
    print(f"Nombre de clusters: {number_cluster}")
    kmeans_hog = KMeans(n_clusters=number_cluster, random_state=42)
    kmeans_hist = KMeans(n_clusters=number_cluster, random_state=42)
    kmeans_resnet = KMeans(n_clusters=number_cluster, random_state=42)
    kmeans_clip = KMeans(n_clusters=number_cluster, random_state=42)

    print("- calcul kmeans avec features HOG (sans supervision) ...")
    kmeans_hog.fit(np.array(descriptors_hog))
    print("- calcul kmeans avec features Histogram (sans supervision)...")
    kmeans_hist.fit(np.array(descriptors_hist))
    print("- calcul kmeans avec descriptors de resnet50...")
    kmeans_resnet.fit(np.array(descriptors_resnet))
    print("- calcul kmeans avec features CLIP (sans supervision)...")
    kmeans_clip.fit(np.array(descriptors_clip))


    print("\n\n ##### Résultat ######")
    metric_hist = show_metric(labels_true, kmeans_hist.labels_, descriptors_hist, bool_show=True, name_descriptor="HISTOGRAM", bool_return=True)
    metric_hog = show_metric(labels_true, kmeans_hog.labels_, descriptors_hog,bool_show=True, name_descriptor="HOG", bool_return=True)
    metric_resnet = show_metric(labels_true, kmeans_resnet.labels_, descriptors_resnet,bool_show=True, name_descriptor="RESNET", bool_return=True)

    metric_clip = show_metric(labels_true, kmeans_clip.labels_, descriptors_clip, bool_show=True, name_descriptor="CLIP", bool_return=True)


    print("- export des données vers le dashboard")
    # conversion des données vers le format du dashboard
    list_dict = [metric_hist,metric_hog,metric_resnet,metric_clip]
    df_metric = pd.DataFrame(list_dict)
    
    # Normalisation des données
    scaler = StandardScaler()
    descriptors_hist_norm = scaler.fit_transform(descriptors_hist)
    descriptors_hog_norm = scaler.fit_transform(descriptors_hog)
    descriptors_resnet_norm = scaler.fit_transform(descriptors_resnet)
    descriptors_clip_norm = scaler.fit_transform(descriptors_clip)

    #conversion vers un format 3D pour la visualisation
    x_3d_hist = conversion_3d(descriptors_hist_norm)
    x_3d_hog = conversion_3d(descriptors_hog_norm)
    x_3d_resnet = conversion_3d(descriptors_resnet_norm)
    x_3d_clip = conversion_3d(descriptors_clip_norm)

    # création des dataframe pour la sauvegarde des données pour la visualisation
    df_resnet = create_df_to_export(x_3d_resnet, labels_true, kmeans_resnet.labels_,image_paths)
    df_hist = create_df_to_export(x_3d_hist, labels_true, kmeans_hist.labels_, image_paths)
    df_hog = create_df_to_export(x_3d_hog, labels_true, kmeans_hog.labels_, image_paths)
    df_clip = create_df_to_export(x_3d_clip, labels_true, kmeans_clip.labels_, image_paths)

    # Vérifie si le dossier existe déjà
    if not os.path.exists(PATH_OUTPUT):
        # Crée le dossier
        os.makedirs(PATH_OUTPUT)

    # sauvegarde des données
    df_hist.to_excel(PATH_OUTPUT+"/save_clustering_hist_kmeans.xlsx")
    df_hog.to_excel(PATH_OUTPUT+"/save_clustering_hog_kmeans.xlsx")
    df_resnet.to_excel(PATH_OUTPUT + "/save_clustering_resnet_kmeans.xlsx")
    df_clip.to_excel(PATH_OUTPUT+"/save_clustering_clip_kmeans.xlsx")
    df_metric.to_excel(PATH_OUTPUT+"/save_metric.xlsx")
    print("Fin. \n\n Pour avoir la visualisation dashboard, veuillez lancer la commande : streamlit run dashboard_clustering.py")


if __name__ == "__main__":
    pipeline()