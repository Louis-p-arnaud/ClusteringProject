import numpy as np
from scipy.linalg import eigh
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
# Tu peux remplacer cet import par ta propre classe KMeans si tu l'as créée
from sklearn.cluster import KMeans 

class SpectralCustom:
    """
    Implémentation "from scratch" de l'algorithme Spectral Clustering.
    """

    def __init__(self, n_clusters=20, n_neighbors=10, random_state=42):
        """
        Initialisation du modèle Spectral Clustering.
        
        Input :
        - n_clusters : (int) nombre de clusters à former.
        - n_neighbors : (int) nombre de voisins pour construire le graphe k-NN.
        - random_state : (int) graine aléatoire pour la reproductibilité.

        Output :
        - None (initialise l'objet)
        """
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.labels = None

    def _construire_matrice_adjacence(self, X):
        """
        Calcule la matrice d'adjacence W basée sur les k plus proches voisins (k-NN).
        
        Input :
        - X : (np.ndarray) tableau des descripteurs de taille (nombre_images, taille_descripteur).
        
        Output :
        - W : (np.ndarray) matrice d'adjacence symétrique de taille (nombre_images, nombre_images).
        """

        #calcule des distances euclidiennes entre toutes les paires de points dans X
        distances = cdist(X,X)

        #calcul des indices des n_neighbors plus proches voisins
        #axis=1 permet de trier ligne par ligne.
        #on prend [:, 1:self.n_neighbors + 1] pour exclure la 1ère colonne (le point lui-même, dont la distance est 0).
        indices_voisins = np.argsort(distances, axis=1)[:, 1:self.n_neighbors + 1]

        #initialisation de la matrice d'adjacence avec une matrice de taille nombre_image*nombre_image
        W = np.zeros((X.shape[0],X.shape[0]))

        #on assigne 1 aux indices correspondants aux n_neighbors voisins pour chaques points
        for i in range (X.shape[0]):
            # On parcourt directement les voisins du point i
            for j in indices_voisins[i]:
                W[i, j] = 1
                W[j, i] = 1  # Règle de symétrie obligatoire pour le Laplacien
        
        return W


    def _calculer_laplacien(self, W):
        """
        Calcule la matrice Laplacienne L = D - W.
        
        Input :
        - W : (np.ndarray) matrice d'adjacence de taille (nombre_images, nombre_images).
        
        Output :
        - L : (np.ndarray) matrice Laplacienne de taille (nombre_images, nombre_images).
        """

        #le degres correspond à la somme de chaque ligne de W
        #axis = 1 afin d'additionner uniquement les lignes
        degres = np.sum(W, axis=1)

        # creation de la matrice diagonale D avec les degrés
        D = np.diag(degres)        
        #calcul du laplacien
        L = D - W

        return L


    def _extraire_vecteurs_propres(self, L):
        """
        Extrait les k premiers vecteurs propres associés aux k plus petites valeurs propres.
        
        Input :
        - L : (np.ndarray) matrice Laplacienne de taille (nombre_images, nombre_images).
        
        Output :
        - vecteurs : (np.ndarray) matrice de taille (nombre_images, n_clusters) contenant les vecteurs propres.
        """
        # calcul des valeurs et vecteurs propres de L
        valeurs_propres, vecteurs_propres = eigh(L)

        #récupération des indices pour trier du plus petit au plus grand
        indices_tri = np.argsort(valeurs_propres)

        #réorganisation des colonnes des vecteurs propres pour suivre cet ordre
        vecteurs_tries = vecteurs_propres[:, indices_tri]

        #découpage de la matrice pour ne garder que les 'n_clusters' premières colonnes
        vecteurs_finaux = vecteurs_tries[:, :self.n_clusters]

        return vecteurs_finaux

    def fit_predict(self, X):
        """
        Orchestre les étapes du Spectral Clustering et retourne les clusters finaux.
        
        Input :
        - X : (np.ndarray) tableau des descripteurs de taille (nombre_images, taille_descripteur).
        
        Output :
        - labels : (np.ndarray) tableau 1D contenant le numéro du cluster pour chaque image.
        """
        # construction de la matrice d'adjacence
        W = self._construire_matrice_adjacence(X)

        # calcul du Laplacien
        L = self._calculer_laplacien(W)

        #extraction des vecteurs propres k premiers vecteurs propres associés aux k plus petites valeurs propres
        vecteurs_propres = self._extraire_vecteurs_propres(L)

        # instanciation de KMeans
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)

        # entraînement sur le nouvel espace spectral (nos vecteurs propres tries) et récupération des labels
        self.labels_ = kmeans.fit_predict(vecteurs_propres)


        return self.labels_