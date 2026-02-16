from email.mime import image
import os
import cv2
import numpy as np
from skimage.feature import hog
from skimage import transform, color
import itertools


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
    
