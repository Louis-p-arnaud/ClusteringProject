from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import pandas as pd
import numpy as np
import torch
import cv2
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from Algos.GMM_algo.GMM_clustering import GaussianMixture
from Algos.dbscan_algo.clustering import show_metric
from Descriptors.ResNet50 import compute_resnet_descriptors
from Descriptors.features import compute_hog_descriptors, compute_color_histograms, compute_clip_descriptors, compute_vit_descriptors
from utils import *
from constant import PATH_ALGO, MODEL_CLUSTERING, PATH_DATASET


def reduce_dimension_for_gmm(X, max_components=50, var_threshold=0.95):
    """
    Réduit la dimension pour stabiliser GMM en haute dimension.
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


def select_best_n_components(X_tensor, n_candidates=None, covariance_type="full", n_iter=100):
    """
    Sélectionne automatiquement le meilleur nombre de composantes GMM via le BIC.
    Un BIC plus bas indique un meilleur modèle (équilibre fit / complexité).
    args:
        X_tensor:         torch.Tensor (n, d)
        n_candidates:     list[int] — nombre de composantes à tester
        covariance_type:  str — "full" ou "diag"
        n_iter:           int — nombre d'itérations EM max
    returns:
        best_n:           int
        best_model:       GaussianMixture
        bic_scores:       dict {n: bic_value}
    """
    if n_candidates is None:
        n_candidates = [2, 3, 4, 5, 6, 8, 10]

    n_samples, n_features = X_tensor.shape
    best_bic = np.inf
    best_n = n_candidates[0]
    best_model = None
    bic_scores = {}

    for n in n_candidates:
        if n >= n_samples:
            continue
        try:
            model = GaussianMixture(
                n_components=n,
                n_features=n_features,
                covariance_type=covariance_type,
            )
            model.fit(X_tensor, n_iter=n_iter)
            bic_val = float(model.bic(X_tensor))
            bic_scores[n] = bic_val

            if bic_val < best_bic:
                best_bic = bic_val
                best_n = n
                best_model = model
        except Exception as e:
            print(f"  [WARN] GMM n_components={n} échoué: {e}")
            bic_scores[n] = np.inf

    return best_n, best_model, bic_scores


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

    dataset_dir = Path(dataset_path)
    label = 0

    for category_folder in sorted(dataset_dir.iterdir()):
        if category_folder.is_dir():
            category_name = category_folder.name
            category_names.append(category_name)

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


def fit_gmm_with_fallback(X_tensor, target_n_components=20, n_iter=100):
    """
    Entraîne GMM avec fallback pour éviter les erreurs de matrice non définie positive.
    Priorité:
    - n_components cible (20)
    - covariance full puis diag
    - si échec, réduction progressive de n_components
    """
    n_samples, n_features = X_tensor.shape

    n_candidates = [target_n_components, 18, 16, 14, 12, 10, 8, 6, 4, 2]
    n_candidates = [n for n in n_candidates if n < n_samples]
    if len(n_candidates) == 0:
        raise RuntimeError("Aucun n_components valide pour GMM.")

    covariance_candidates = ["full", "diag"]
    last_error = None

    for covariance_type in covariance_candidates:
        for n_components in n_candidates:
            try:
                model = GaussianMixture(
                    n_components=n_components,
                    n_features=n_features,
                    covariance_type=covariance_type,
                )
                model.fit(X_tensor, n_iter=n_iter)
                bic_val = float(model.bic(X_tensor))
                return model, n_components, covariance_type, bic_val
            except Exception as exc:
                last_error = exc
                print(
                    f"  [WARN] Echec GMM (covariance={covariance_type}, n_components={n_components}): {exc}"
                )

    raise RuntimeError(f"Impossible d'entraîner GMM après fallback. Dernière erreur: {last_error}")


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

    print("\n\n ##### Clustering GMM (NON SUPERVISÉ) ######")
    target_n_components = 20
    print(f"Nombre de composantes fixé à {target_n_components}")

    metrics_list = []
    labels_by_descriptor = {}
    features_for_viz = {}

    for descriptor_name, descriptor_values in descriptor_map.items():
        print(f"\n--- GMM {descriptor_name} ---")

        # 1) Standardisation
        X_scaled = StandardScaler().fit_transform(descriptor_values)

        # 2) Réduction de dimension pour stabiliser les distances en haute dimension
        X_cluster, pca_dims = reduce_dimension_for_gmm(X_scaled, max_components=30, var_threshold=0.9)

        # 3) Conversion en tensor PyTorch
        X_tensor = torch.FloatTensor(X_cluster)
        n_features = X_tensor.shape[1]

        # 4) Entraînement GMM robuste avec fallback (full -> diag, puis baisse n_components)
        best_model, best_n, covariance_type, bic_val = fit_gmm_with_fallback(
            X_tensor,
            target_n_components=target_n_components,
            n_iter=100,
        )
        bic_scores = {best_n: bic_val}

        print(f"BIC {descriptor_name}: k={best_n}, covariance={covariance_type}: {bic_val:.1f}")
        print(
            f"Paramètres {descriptor_name}: n_components={best_n}, "
            f"dims={X_cluster.shape[1]} (PCA={pca_dims})"
        )

        # 5) Prédiction des labels de clustering
        labels_pred = best_model.predict(X_tensor).numpy()
        n_clusters = len(np.unique(labels_pred))

        print(f"Résultat {descriptor_name}: clusters trouvés={n_clusters}")

        # 6) Métriques (utilise labels_true uniquement pour évaluation post-clustering)
        metric = show_metric(
            labels_true,
            labels_pred,
            X_cluster,
            bool_show=True,
            name_descriptor=descriptor_name,
            bool_return=True,
            name_model="gmm",
        )
        metric["n_components"] = best_n
        metric["covariance_type"] = covariance_type
        metric["n_clusters"] = n_clusters
        metric["bic_scores"] = str(bic_scores)
        metrics_list.append(metric)

        labels_by_descriptor[descriptor_name] = labels_pred
        features_for_viz[descriptor_name] = X_cluster

    print("- export des données vers le dashboard")
    # Conversion des données vers le format du dashboard
    df_metric = pd.DataFrame(metrics_list)

    # Conversion vers un format 3D pour la visualisation
    x_3d_hist = conversion_3d(features_for_viz["HISTOGRAM"])
    x_3d_hog = conversion_3d(features_for_viz["HOG"])
    x_3d_resnet = conversion_3d(features_for_viz["RESNET"])
    x_3d_clip = conversion_3d(features_for_viz["CLIP"])
    x_3d_vit = conversion_3d(features_for_viz["VIT"])

    # Création des dataframes pour la sauvegarde des données pour la visualisation
    df_hist = create_df_to_export(x_3d_hist, labels_true, labels_by_descriptor["HISTOGRAM"], image_paths)
    df_hog = create_df_to_export(x_3d_hog, labels_true, labels_by_descriptor["HOG"], image_paths)
    df_resnet = create_df_to_export(x_3d_resnet, labels_true, labels_by_descriptor["RESNET"], image_paths)
    df_clip = create_df_to_export(x_3d_clip, labels_true, labels_by_descriptor["CLIP"], image_paths)
    df_vit = create_df_to_export(x_3d_vit, labels_true, labels_by_descriptor["VIT"], image_paths)

    # Chemin de sortie GMM
    PATH_OUTPUT_GMM = os.path.join(PATH_ALGO, "GMM_algo", "output")

    # Vérifie si le dossier existe déjà
    if not os.path.exists(PATH_OUTPUT_GMM):
        os.makedirs(PATH_OUTPUT_GMM)

    # Sauvegarde des données
    df_hist.to_excel(PATH_OUTPUT_GMM + "/save_clustering_hist_gmm.xlsx")
    df_hog.to_excel(PATH_OUTPUT_GMM + "/save_clustering_hog_gmm.xlsx")
    df_resnet.to_excel(PATH_OUTPUT_GMM + "/save_clustering_resnet_gmm.xlsx")
    df_clip.to_excel(PATH_OUTPUT_GMM + "/save_clustering_clip_gmm.xlsx")
    df_vit.to_excel(PATH_OUTPUT_GMM + "/save_clustering_vit_gmm.xlsx")
    df_metric.to_excel(PATH_OUTPUT_GMM + "/save_metric.xlsx")
    print("Fin. \n\n Pour avoir la visualisation dashboard, veuillez lancer la commande : streamlit run dashboard_clustering.py")


if __name__ == "__main__":
    pipeline()
