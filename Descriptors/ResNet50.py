import torch
import numpy as np
import cv2
from torchvision import models, transforms
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def compute_resnet_descriptors(images, batch_size=32):
    """
    Extrait les features 2048D de ResNet50 pour une liste d'images OpenCV.
    """
    # 1. Configuration hardware et modèle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #IMAGENET1K_V1 : Nom du jeu de données (ImageNet) contenant plus d'un million d'images réparties en 1000 catégories
    #=> Le modèle est préentrainé
    model = models.resnet50(weights='IMAGENET1K_V1')
    model.fc = torch.nn.Identity()  # On garde les features, pas la classification
    model.to(device)
    model.eval()

    # 2. Prétraitement (ResNet attend du RGB 224x224 normalisé)
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    descriptors = []

    # 3. Extraction par batchs
    for i in range(0, len(images), batch_size):
        batch_list = images[i:i + batch_size]

        # Conversion BGR (OpenCV) vers RGB et mise en tenseur
        batch_tensors = torch.stack([preprocess(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in batch_list])
        batch_tensors = batch_tensors.to(device)

        with torch.no_grad():
            features = model(batch_tensors)
            descriptors.append(features.cpu().numpy())

    return np.concatenate(descriptors, axis=0)


def prepare_for_clustering(features, n_components=50):
    """
    Normalise et réduit la dimensionnalité pour optimiser le clustering.
    """
    # Normalisation (moyenne 0, variance 1)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Réduction de dimension (PCA)
    pca = PCA(n_components=n_components)
    features_reduced = pca.fit_transform(features_scaled)

    return features_reduced

# --- EXEMPLE D'UTILISATION ---
# images_list = [cv2.imread(path) for path in mes_chemins]
# raw_features = compute_resnet_descriptors(images_list)
# final_descriptors = prepare_for_clustering(raw_features)