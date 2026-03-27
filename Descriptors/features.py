from email.mime import image
import os
import cv2
import numpy as np
from skimage.feature import hog
from skimage import transform, color
import itertools


def compute_vit_descriptors(images,
                            model_name='vit_tiny_patch16_224',
                            batch_size=32,
                            device=None):
    """
    Calcule les descripteurs Vision Transformer (ViT) pour les images.
    Input : images (array) : tableau numpy des images (BGR ou RGB)
    Output : descriptors (np.array) : matrice des descripteurs ViT (N, D)
    """
    
    # try/except évite de faire planter tout le pipeline si une librairie manque sur une machine
    try:
        import torch
        import timm
        from PIL import Image
        from torchvision import transforms
    except ImportError as exc:
        raise ImportError(
            "ViT nécessite les packages 'torch', 'timm', 'Pillow' et 'torchvision'. "
            "Installe-les puis relance le pipeline."
        ) from exc

    
    #permet d'utiliser CUDA si dispo, sinon on bascule sur notre processeur (CPU)
    #on prefere utiliser CUDA car Calcul en Parallèle > Calcul Séquentiel
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # chargement du modèle ViT
    # on obtient directement le vecteur de caractéristiques (le descripteur) au lieu d'une prédiction
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.to(device)
    model.eval() # le modele passe en mode evaluation (et n'est plus en mode entrainement ducoup) => resultat verrouille

    # phase de preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # Le ViT n'accepte que des images de 224 pixels de cotes
        transforms.ToTensor(),         # Conversion au format Tensor PyTorch (les valeurs sont ramenes entre 0 et 1)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    descriptors = []
    images = np.array(images)

    # traitement par lots (batching)
    # On découpe les images par petits groupes (batch_size=32) pour ne pas saturer la RAM
    for start_idx in range(0, len(images), batch_size):
        batch = images[start_idx:start_idx + batch_size]
        batch_tensors = []

        # Gestion robuste des canaux de couleurs
        # On s'assure de toujours envoyer du RGB au modèle, peu importe le format d'origine.
        for img in batch:
            if len(img.shape) == 2: # Image en noir et blanc (1 canal)
                rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif len(img.shape) == 3 and img.shape[2] == 3: # Image couleur standard OpenCV (BGR)
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif len(img.shape) == 3 and img.shape[2] == 4: # Image avec transparence (RGBA)
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            else:
                rgb_img = img # Fallback

            pil_img = Image.fromarray(rgb_img)
            batch_tensors.append(transform(pil_img))

        # Envoi du batch préparé vers CUDA/CPU
        batch_tensors = torch.stack(batch_tensors).to(device)

        # Inférence et Normalisation
        # torch.no_grad() coupe le calcul des gradients, économisant énormément de mémoire et de temps
        # le mot cle with avec torch.no_grad permet donc de forcer PyTorch à ne pas calculer de gradients pendant l'extraction (opti pour la consommation de RAM)
        with torch.no_grad():
            features = model(batch_tensors) # Extraction du vecteur par le ViT
            
            # Normalisation L2 : on divise le vecteur par sa norme.
            # Très important pour K-Means/Spectral car cela transforme les distances en similarités d'angles.
            norm = features.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
            features = features / norm

        # On ramène les descripteurs sur le processeur (CPU) pour les stocker
        descriptors.append(features.cpu().numpy())

    # Assemblage final
    # On fusionne tous les sous-tableaux en une seule grande matrice (N images, D dimensions)
    return np.vstack(descriptors).astype(np.float32)

def compute_clip_descriptors(images,
                             model_name="openai/clip-vit-base-patch32",
                             batch_size=32,
                             device=None):
    """
    Calcule les descripteurs CLIP pour les images.
    Input : images (array) : tableau numpy des images (BGR ou RGB)
    Output : descriptors (np.array) : matrice des descripteurs CLIP (N, D)
    """
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise ImportError(
            "CLIP nécessite les packages 'torch' et 'transformers'. "
            "Installe-les puis relance le pipeline."
        ) from exc

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.to(device)
    model.eval()

    descriptors = []
    images = np.array(images)

    for start_idx in range(0, len(images), batch_size):
        batch = images[start_idx:start_idx + batch_size]
        batch_rgb = []
        for img in batch:
            if len(img.shape) == 3 and img.shape[2] == 3:
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            batch_rgb.append(rgb_img)

        inputs = processor(images=batch_rgb, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        with torch.no_grad():
            batch_features = model.get_image_features(pixel_values=pixel_values)

            if hasattr(batch_features, "image_embeds"):
                batch_features = batch_features.image_embeds
            elif hasattr(batch_features, "pooler_output"):
                pooled_features = batch_features.pooler_output
                if hasattr(model, "visual_projection") and model.visual_projection is not None:
                    in_features = model.visual_projection.in_features
                    if pooled_features.shape[-1] == in_features:
                        batch_features = model.visual_projection(pooled_features)
                    else:
                        batch_features = pooled_features
                else:
                    batch_features = pooled_features
            elif isinstance(batch_features, (tuple, list)):
                batch_features = batch_features[0]

            if not torch.is_tensor(batch_features):
                raise TypeError("Sortie CLIP inattendue: impossible de convertir en tenseur de features image.")

            norm = batch_features.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
            batch_features = batch_features / norm

        descriptors.append(batch_features.cpu().numpy())

    return np.vstack(descriptors).astype(np.float32)


def compute_gray_histograms(images):
    """
    Calcule les histogrammes de niveau de gris pour les images.
    Input : images (array) : tableau numpy des images (BGR ou niveaux de gris)
    Output : descriptors (list) : liste des descripteurs d'histogrammes de niveau de gris
    """
    descriptors = []

    for img in images:
        # Convertir en niveaux de gris si l'image est en couleur
        if len(img.shape) == 3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = img
        
        gray_img = gray_img.astype(np.float32)
        hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        descriptors.append(hist)

    return descriptors


def compute_color_histograms(images):
    """
    Calcule les histogrammes de couleur HSV pour les images.
    Input : images (array) : tableau numpy des images en couleur (BGR)
    Output : descriptors (list) : liste des descripteurs d'histogrammes de couleur
    """
    descriptors = []

    for img in images:
        # Convertir BGR vers HSV
        if len(img.shape) == 3 and img.shape[2] == 3:
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Calcul des histogrammes pour chaque canal HSV
            h_hist = cv2.calcHist([hsv_img], [0], None, [180], [0, 180])
            s_hist = cv2.calcHist([hsv_img], [1], None, [256], [0, 256])
            v_hist = cv2.calcHist([hsv_img], [2], None, [256], [0, 256])
            
            # Normalisation
            h_hist = cv2.normalize(h_hist, h_hist).flatten()
            s_hist = cv2.normalize(s_hist, s_hist).flatten()
            v_hist = cv2.normalize(v_hist, v_hist).flatten()
            
            # Concaténation des histogrammes
            color_hist = np.concatenate([h_hist, s_hist, v_hist])
            descriptors.append(color_hist)
        else:
            # Si image en niveaux de gris
            gray_img = img.astype(np.float32) if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
            hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            descriptors.append(hist)

    return descriptors


def compute_hog_descriptors(images):
    """
    Calcule les descripteurs HOG pour les images.
    Input : images (array) : tableau numpy des images (BGR ou niveaux de gris)
    Output : descriptors (list) : liste des descripteurs HOG
    """
    descriptors = []
    
    for img in images:
        # Convertir en niveaux de gris si l'image est en couleur
        if len(img.shape) == 3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = img
        
        fd, hog_image = hog(
            gray_img,
            orientations=8,
            pixels_per_cell=(4, 4),
            cells_per_block=(1, 1),
            visualize=True,
            channel_axis=None,
        )

        descriptors.append(fd)

    return descriptors
    
