from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Algos.spectral_clustering_algo.spectral_clustering_from_scratch import SpectralClustering
from Algos.kmeans_algo.clustering import show_metric
from Descriptors.ResNet50 import compute_resnet_descriptors
from Descriptors.features import compute_hog_descriptors, compute_color_histograms, compute_clip_descriptors, compute_vit_descriptors
from utils import *
from constant import PATH_ALGO, PATH_DATASET


def load_images_from_dataset(dataset_path):
    """
    Charge les images depuis le dossier dataset sans utiliser les noms des dossiers pour l'apprentissage.
    """
    images = []
    labels_true = []
    category_names = []
    image_paths = []

    dataset_dir = Path(dataset_path)
    label = 0

    for category_folder in sorted(dataset_dir.iterdir()):
        if category_folder.is_dir():
            category_names.append(category_folder.name)

            for img_file in sorted(category_folder.glob("*")):
                if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                    try:
                        with open(img_file, 'rb') as f:
                            img_bytes = np.frombuffer(f.read(), np.uint8)
                        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

                        if img is not None:
                            img = cv2.resize(img, (64, 64))
                            images.append(img)
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
    print("- calcul features CLIP...")
    descriptors_clip = compute_clip_descriptors(images)
    print("- calcul features ViT...")
    descriptors_vit = compute_vit_descriptors(images)

    descriptor_map = {
        "HOG": np.asarray(descriptors_hog),
        "HISTOGRAM": np.asarray(descriptors_hist),
        "RESNET": np.asarray(descriptors_resnet),
        "CLIP": np.asarray(descriptors_clip),
        "VIT": np.asarray(descriptors_vit),
    }

    print("\n\n ##### Clustering Spectral (NON SUPERVISÉ) ######")
    number_cluster = 20
    print(f"Nombre de clusters: {number_cluster}")

    labels_by_descriptor = {}
    metrics_list = []

    for descriptor_name, descriptor_values in descriptor_map.items():
        spectral_model = SpectralClustering(
            n_clusters=number_cluster,
            n_neighbors=10,
            random_state=42,
        )

        print(f"- calcul spectral clustering avec features {descriptor_name} (sans supervision)...")
        spectral_model.fit_predict(descriptor_values)
        labels_by_descriptor[descriptor_name] = spectral_model.labels_

        metric = show_metric(
            labels_true,
            spectral_model.labels_,
            descriptor_values,
            bool_show=True,
            name_descriptor=descriptor_name,
            bool_return=True,
            name_model="spectral",
        )
        metrics_list.append(metric)

    print("\n\n ##### Résultat ######")
    print("- export des données vers le dashboard")
    df_metric = pd.DataFrame(metrics_list)

    descriptors_norm = {}
    for descriptor_name, descriptor_values in descriptor_map.items():
        descriptors_norm[descriptor_name] = StandardScaler().fit_transform(descriptor_values)

    x_3d_hist = conversion_3d(descriptors_norm["HISTOGRAM"])
    x_3d_hog = conversion_3d(descriptors_norm["HOG"])
    x_3d_resnet = conversion_3d(descriptors_norm["RESNET"])
    x_3d_clip = conversion_3d(descriptors_norm["CLIP"])
    x_3d_vit = conversion_3d(descriptors_norm["VIT"])

    df_hist = create_df_to_export(x_3d_hist, labels_true, labels_by_descriptor["HISTOGRAM"], image_paths)
    df_hog = create_df_to_export(x_3d_hog, labels_true, labels_by_descriptor["HOG"], image_paths)
    df_resnet = create_df_to_export(x_3d_resnet, labels_true, labels_by_descriptor["RESNET"], image_paths)
    df_clip = create_df_to_export(x_3d_clip, labels_true, labels_by_descriptor["CLIP"], image_paths)
    df_vit = create_df_to_export(x_3d_vit, labels_true, labels_by_descriptor["VIT"], image_paths)

    path_output_spectral = os.path.join(PATH_ALGO, "spectral_clustering_algo", "output")
    if not os.path.exists(path_output_spectral):
        os.makedirs(path_output_spectral)

    df_hist.to_excel(path_output_spectral + "/save_clustering_hist_spectral.xlsx")
    df_hog.to_excel(path_output_spectral + "/save_clustering_hog_spectral.xlsx")
    df_resnet.to_excel(path_output_spectral + "/save_clustering_resnet_spectral.xlsx")
    df_clip.to_excel(path_output_spectral + "/save_clustering_clip_spectral.xlsx")
    df_vit.to_excel(path_output_spectral + "/save_clustering_vit_spectral.xlsx")
    df_metric.to_excel(path_output_spectral + "/save_metric.xlsx")

    print("Fin. \n\n Pour avoir la visualisation dashboard, veuillez lancer la commande : streamlit run dashboard_clustering.py")


if __name__ == "__main__":
    pipeline()
