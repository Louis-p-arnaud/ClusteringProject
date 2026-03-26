import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import eigh
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
        pass

    def _construire_matrice_adjacence(self, X):
        """
        Calcule la matrice d'adjacence W basée sur les k plus proches voisins (k-NN).
        
        Input :
        - X : (np.ndarray) tableau des descripteurs de taille (nombre_images, taille_descripteur).
        
        Output :
        - W : (np.ndarray) matrice d'adjacence symétrique de taille (nombre_images, nombre_images).
        """
        # TODO : 1. Calculer les distances entre toutes les paires de points dans X (utiliser cdist).
        # TODO : 2. Pour chaque point, trouver les indices de ses 'n_neighbors' plus proches voisins.
        # TODO : 3. Créer une matrice W remplie de 0.
        # TODO : 4. Placer des 1 dans W pour chaque lien voisin trouvé.
        # TODO : 5. Rendre la matrice W symétrique (si A est voisin de B, B est voisin de A) et retourner W.
        pass

    def _calculer_laplacien(self, W):
        """
        Calcule la matrice Laplacienne L = D - W.
        
        Input :
        - W : (np.ndarray) matrice d'adjacence de taille (nombre_images, nombre_images).
        
        Output :
        - L : (np.ndarray) matrice Laplacienne de taille (nombre_images, nombre_images).
        """
        # TODO : 1. Calculer le degré de chaque nœud (somme de chaque ligne de W).
        # TODO : 2. Créer la matrice diagonale D avec ces degrés (utiliser np.diag).
        # TODO : 3. Calculer L = D - W et retourner L.
        pass

    def _extraire_vecteurs_propres(self, L):
        """
        Extrait les k premiers vecteurs propres associés aux k plus petites valeurs propres.
        
        Input :
        - L : (np.ndarray) matrice Laplacienne de taille (nombre_images, nombre_images).
        
        Output :
        - vecteurs : (np.ndarray) matrice de taille (nombre_images, n_clusters) contenant les vecteurs propres.
        """
        # TODO : 1. Calculer les valeurs propres et vecteurs propres de L (utiliser eigh de scipy).
        # TODO : 2. Trier les valeurs propres par ordre croissant pour obtenir les indices de tri.
        # TODO : 3. Trier les vecteurs propres en utilisant ces mêmes indices.
        # TODO : 4. Sélectionner et retourner uniquement les 'n_clusters' premiers vecteurs propres.
        pass

    def fit_predict(self, X):
        """
        Orchestre les étapes du Spectral Clustering et retourne les clusters finaux.
        
        Input :
        - X : (np.ndarray) tableau des descripteurs de taille (nombre_images, taille_descripteur).
        
        Output :
        - labels : (np.ndarray) tableau 1D contenant le numéro du cluster pour chaque image.
        """
        # TODO : 1. W = self._construire_matrice_adjacence(X)
        # TODO : 2. L = self._calculer_laplacien(W)
        # TODO : 3. vecteurs_propres = self._extraire_vecteurs_propres(L)
        # TODO : 4. Instancier KMeans avec self.n_clusters et self.random_state.
        # TODO : 5. Entraîner le KMeans sur 'vecteurs_propres' et retourner les labels obtenus.
        pass