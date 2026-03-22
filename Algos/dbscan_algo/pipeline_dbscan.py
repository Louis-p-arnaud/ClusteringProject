from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import os
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Algos.dbscan_algo.clustering import DBSCAN, show_metric
from Descriptors.ResNet50 import compute_resnet_descriptors, prepare_for_clustering
from Descriptors.features import compute_hog_descriptors, compute_color_histograms, compute_clip_descriptors, compute_vit_descriptors
from utils import *
from constant import PATH_ALGO, MODEL_CLUSTERING, PATH_DATASET


def reduce_dimension_for_dbscan(X, max_components=50, var_threshold=0.95):
    """
    Réduit la dimension pour stabiliser DBSCAN en haute dimension.
    Retourne les données réduites et le nombre de composantes retenues.
    """
    X = np.asarray(X)
    n_samples, n_features = X.shape

    if n_features <= max_components:
        return X, n_features

    n_components = min(max_components, n_samples - 1, n_features)
    if n_components < 2:
        return X, n_features

    pca = PCA(n_components=n_components, random_state=42)
    X_reduced = pca.fit_transform(X)

    explained = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(explained, var_threshold) + 1)
    k = max(2, min(k, X_reduced.shape[1]))
    return X_reduced[:, :k], k


def estimate_eps(X, min_samples=5, quantile=0.92):
    """
    Estime eps via la k-distance (distance au min_samples-ieme voisin).
    """
    X = np.asarray(X)
    n_samples = X.shape[0]
    n_neighbors = max(2, min(min_samples, n_samples))

    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)

    kth_distances = np.sort(distances[:, -1])
    eps = float(np.quantile(kth_distances, quantile))

    if eps <= 0:
        eps = float(np.median(kth_distances))
    if eps <= 0:
        eps = 0.5

    return eps


def count_valid_clusters(labels):
    labels = np.asarray(labels)
    non_noise = labels[labels != -1]
    if len(non_noise) == 0:
        return 0
    return int(len(np.unique(non_noise)))


def tune_dbscan_params(X, quantiles=None, min_samples_candidates=None):
    """
    Cherche automatiquement un bon couple (eps, min_samples) pour éviter
    le cas "1 cluster geant" ou "tout en bruit".
    """
    if quantiles is None:
        quantiles = [0.55, 0.65, 0.72, 0.8, 0.88]
    if min_samples_candidates is None:
        min_samples_candidates = [3, 5, 8]

    best_valid = None
    best_any = None

    for min_samples_value in min_samples_candidates:
        for q in quantiles:
            eps_value = estimate_eps(X, min_samples=min_samples_value, quantile=q)
            model = DBSCAN(eps=eps_value, min_samples=min_samples_value)
            model.fit(X)
            labels_pred = model.labels_

            n_clusters = count_valid_clusters(labels_pred)
            noise_ratio = float(np.mean(labels_pred == -1))

            candidate = {
                "labels": labels_pred,
                "eps": eps_value,
                "min_samples": min_samples_value,
                "quantile": q,
                "n_clusters": n_clusters,
                "noise_ratio": noise_ratio,
            }

            # Priorite: au moins 2 clusters et un niveau de bruit raisonnable.
            # Ensuite, maximiser le nombre de clusters sans exploser le bruit.
            if n_clusters >= 2 and noise_ratio <= 0.8:
                score = n_clusters - (noise_ratio * 5.0)
                candidate["score"] = score
                if best_valid is None or score > best_valid["score"]:
                    best_valid = candidate

            # Fallback: garder la meilleure config meme si invalide
            fallback_score = (n_clusters * 2.0) - (noise_ratio * 3.0)
            candidate["score"] = fallback_score
            if best_any is None or fallback_score > best_any["score"]:
                best_any = candidate

    return best_valid if best_valid is not None else best_any


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

    print("\n\n ##### Clustering DBSCAN (NON SUPERVISÉ) ######")
    print("Auto-tuning par descripteur: recherche de eps et min_samples")

    metrics_list = []
    labels_by_descriptor = {}
    features_for_viz = {}

    for descriptor_name, descriptor_values in descriptor_map.items():
        print(f"\n--- DBSCAN {descriptor_name} ---")

        # 1) Standardisation
        X_scaled = StandardScaler().fit_transform(descriptor_values)

        # 2) Réduction de dimension pour stabiliser les distances
        X_cluster, pca_dims = reduce_dimension_for_dbscan(X_scaled, max_components=30, var_threshold=0.9)

        # 3) Auto-tuning DBSCAN
        tuned = tune_dbscan_params(
            X_cluster,
            quantiles=[0.55, 0.65, 0.72, 0.8, 0.88],
            min_samples_candidates=[3, 5, 8],
        )

        labels_pred = tuned["labels"]
        eps_value = tuned["eps"]
        min_samples_value = tuned["min_samples"]
        n_clusters = tuned["n_clusters"]
        noise_ratio = tuned["noise_ratio"]
        noise_count = int(np.sum(labels_pred == -1))

        print(
            f"Paramètres {descriptor_name}: eps={eps_value:.4f}, min_samples={min_samples_value}, "
            f"quantile={tuned['quantile']}, dims={X_cluster.shape[1]} (PCA={pca_dims})"
        )
        print(f"Résultat {descriptor_name}: clusters={n_clusters}, bruit={noise_count}/{len(labels_pred)} ({noise_ratio:.1%})")

        # 5) Métriques
        metric = show_metric(
            labels_true,
            labels_pred,
            X_cluster,
            bool_show=True,
            name_descriptor=descriptor_name,
            bool_return=True,
            name_model="dbscan",
        )
        metric["eps"] = eps_value
        metric["min_samples"] = min_samples_value
        metric["eps_quantile"] = tuned["quantile"]
        metric["n_clusters"] = n_clusters
        metric["noise_ratio"] = noise_ratio
        metrics_list.append(metric)

        labels_by_descriptor[descriptor_name] = labels_pred
        features_for_viz[descriptor_name] = X_cluster

    print("- export des données vers le dashboard")
    # conversion des données vers le format du dashboard
    df_metric = pd.DataFrame(metrics_list)
    
    # conversion vers un format 3D pour la visualisation
    x_3d_hist = conversion_3d(features_for_viz["HISTOGRAM"])
    x_3d_hog = conversion_3d(features_for_viz["HOG"])
    x_3d_resnet = conversion_3d(features_for_viz["RESNET"])
    x_3d_clip = conversion_3d(features_for_viz["CLIP"])
    x_3d_vit = conversion_3d(features_for_viz["VIT"])

    # création des dataframe pour la sauvegarde des données pour la visualisation
    df_hist = create_df_to_export(x_3d_hist, labels_true, labels_by_descriptor["HISTOGRAM"], image_paths)
    df_hog = create_df_to_export(x_3d_hog, labels_true, labels_by_descriptor["HOG"], image_paths)
    df_resnet = create_df_to_export(x_3d_resnet, labels_true, labels_by_descriptor["RESNET"], image_paths)
    df_clip = create_df_to_export(x_3d_clip, labels_true, labels_by_descriptor["CLIP"], image_paths)
    df_vit = create_df_to_export(x_3d_vit, labels_true, labels_by_descriptor["VIT"], image_paths)

    # Chemin de sortie DBSCAN
    PATH_OUTPUT_DBSCAN = os.path.join(PATH_ALGO, "dbscan_algo", "output")
    
    # Vérifie si le dossier existe déjà
    if not os.path.exists(PATH_OUTPUT_DBSCAN):
        os.makedirs(PATH_OUTPUT_DBSCAN)

    # sauvegarde des données
    df_hist.to_excel(PATH_OUTPUT_DBSCAN + "/save_clustering_hist_dbscan.xlsx")
    df_hog.to_excel(PATH_OUTPUT_DBSCAN + "/save_clustering_hog_dbscan.xlsx")
    df_resnet.to_excel(PATH_OUTPUT_DBSCAN + "/save_clustering_resnet_dbscan.xlsx")
    df_clip.to_excel(PATH_OUTPUT_DBSCAN + "/save_clustering_clip_dbscan.xlsx")
    df_vit.to_excel(PATH_OUTPUT_DBSCAN + "/save_clustering_vit_dbscan.xlsx")
    df_metric.to_excel(PATH_OUTPUT_DBSCAN + "/save_metric.xlsx")
    print("Fin. \n\n Pour avoir la visualisation dashboard, veuillez lancer la commande : streamlit run dashboard_clustering.py")


if __name__ == "__main__":
    pipeline()
