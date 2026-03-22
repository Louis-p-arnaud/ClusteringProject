from sklearn.cluster import SpectralClustering

class Spectral:
    """
    Classe encapsulant l'algorithme Spectral Clustering de scikit-learn.
    """

    def __init__(self, n_clusters=20, affinity='nearest_neighbors', assign_labels='kmeans', random_state=42):
        """
        Initialisation du modèle de Spectral Clustering.
        
        Input :
        - n_clusters : (int) nombre de clusters (groupes) à former.
        - affinity : (str) méthode pour construire la matrice d'adjacence (ex: 'nearest_neighbors').
        - assign_labels : (str) algorithme utilisé pour l'étape finale d'assignation (ex: 'kmeans').
        - random_state : (int) graine aléatoire pour garantir la reproductibilité des résultats.

        Output :
        - None (initialise l'objet)
        """
        # TODO : Sauvegarder les paramètres dans les attributs de l'objet (self)
        # TODO : Instancier le modèle SpectralClustering de sklearn avec ces paramètres
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.assign_labels = assign_labels
        self.random_state = random_state
        
        self.model = SpectralClustering(n_clusters=self.n_clusters, affinity=self.affinity, assign_labels=self.assign_labels, random_state=self.random_state)

        pass

    def fit_predict(self, X):
        """
        Applique l'algorithme Spectral Clustering sur les données fournies et retourne les prédictions.
        
        Input :
        - X : (np.ndarray ou list) tableau contenant les descripteurs des images (ex: features ViT). 
              Forme attendue : (nombre_images, taille_descripteur).

        Output :
        - labels : (np.ndarray) tableau à 1 dimension contenant le numéro du cluster attribué à chaque image.
        """
        # TODO : Utiliser le modèle instancié dans __init__ pour entrainer et prédire sur X
        # TODO : Retourner les labels générés

        labels = self.model.fit_predict(X)
        return labels
        

    
