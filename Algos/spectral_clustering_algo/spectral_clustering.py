from sklearn.cluster import SpectralClustering


class Spectral:
    """
    Classe encapsulant l'algorithme Spectral Clustering de scikit-learn.
    """

    def __init__(self, n_clusters=20, affinity='nearest_neighbors', assign_labels='kmeans', random_state=42):
        """
        Initialisation du modèle de Spectral Clustering.
        """
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.assign_labels = assign_labels
        self.random_state = random_state
        self.labels_ = None

        self.model = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity=self.affinity,
            assign_labels=self.assign_labels,
            random_state=self.random_state,
        )

    def fit(self, X):
        """
        Entraîne le modèle Spectral Clustering et stocke les labels.
        """
        self.labels_ = self.model.fit_predict(X)

    def fit_predict(self, X):
        """
        Entraîne le modèle Spectral Clustering et retourne les labels.
        """
        self.fit(X)
        return self.labels_
