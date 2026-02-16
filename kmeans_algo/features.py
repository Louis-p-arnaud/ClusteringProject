from email.mime import image
import os
import cv2
import numpy as np
from skimage.feature import hog
from skimage import transform
import itertools


def compute_gray_histograms(images):
    """
    Calcule les histogrammes de niveau de gris pour les images MNIST.
    Input : images (list) : liste des images en niveaux de gris
    Output : descriptors (list) : liste des descripteurs d'histogrammes de niveau de gris
    """
    descriptors = []

    ##VERSION cv2.calcHist (ne fonctionne pas )
    for img in images:
        img = img.astype(np.float32)
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        descriptors.append(hist)
    
    #VERSION np.histogram
    # for img in images:
    #     # Aplatir l'image en 1D
    #     img_flat = img.flatten()
        
    #     # Calcul de l'histogramme avec numpy (équivalent à cv2.calcHist)
    #     hist, _ = np.histogram(img_flat, bins=256, range=(0, 256))
        
    #     # Convertir en float pour normalisation
    #     hist = hist.astype(np.float32)
        
    #     descriptors.append(hist)

    return descriptors

def compute_hog_descriptors(images):
    """
    Calcule les descripteurs HOG pour les images en niveaux de gris.
    Input : images (array) : tableau numpy des images
    Output : descriptors (list) : liste des descripteurs HOG
    """
    descriptors = []
    
    for img in images:
        fd, hog_image = hog(
            img,
            orientations=8,
            pixels_per_cell=(4, 4),
            cells_per_block=(1, 1),
            visualize=True,
            channel_axis=None,
        )

        descriptors.append(fd)

    return descriptors
    
