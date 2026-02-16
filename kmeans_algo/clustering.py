from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import numpy as np
from sklearn import metrics


class KMeans:
    def __init__(self, n_clusters=8, max_iter=300, random_state=None):
        """
        Initialise un objet KMeans.

        Entrées:
        - n_clusters (int): Le nombre de clusters à former (par défaut 8).
        - max_iter (int): Le nombre maximum d'itérations pour l'algorithme (par défaut 300).
        - random_state (int ou None): La graine pour initialiser le générateur de nombres aléatoires (par défaut None).
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

    def initialize_centers(self, X):
        """
        Initialise les centres de clusters avec n_clusters points choisis aléatoirement à partir des données X.

        Entrée:
        - X (np.array): Les données d'entrée.

        Sortie:
        - Aucune sortie directe, mais les centres de clusters sont stockés dans self.cluster_centers_.
        """
        # Si random_state est défini, on fixe la graine pour la reproductibilité
        # (pour avoir les mêmes résultats à chaque exécution)
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Choisir aléatoirement n_clusters indices parmi les données
        # np.random.choice sélectionne des indices aléatoires sans remplacement
        # X.shape[0] donne le nombre total de lignes (points) dans X
        random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        
        # Sélectionner les points correspondants comme centres initiaux
        # X[random_indices] récupère les lignes aux indices choisis
        self.cluster_centers_ = X[random_indices]

    def nearest_cluster(self, X):
        """
        Calcule la distance euclidienne entre chaque point de X et les centres de clusters,
        puis retourne l'indice du cluster le plus proche pour chaque point.

        Entrée:
        - X (np.array): Les données d'entrée.

        Sortie:
        - np.array: Un tableau d'indices représentant le cluster le plus proche pour chaque point.
        """
        # Initialiser un tableau pour stocker les distances
        # Shape: (nombre de points, nombre de clusters)
        distances = np.zeros((X.shape[0], self.n_clusters))
        
        # Pour chaque centre de cluster
        for i in range(self.n_clusters):
            # Calculer la distance euclidienne entre tous les points et ce centre
            # X - self.cluster_centers_[i] : différence entre chaque point et le centre
            distances[:, i] = np.sqrt(np.sum((X - self.cluster_centers_[i]) ** 2, axis=1))
        
        # Pour chaque point, trouver l'indice du cluster le plus proche (distance minimale)
        # np.argmin(axis=1) : retourne l'indice de la valeur minimale pour chaque ligne
        return np.argmin(distances, axis=1)

    def fit(self, X):
        """
        Exécute l'algorithme K-means sur les données X.

        Entrée:
        - X (np.array): Les données d'entrée.

        Sortie:
        - Aucune sortie directe, mais les centres de clusters sont stockés dans self.cluster_centers_.
        """
        # Étape 1 : Initialiser les centres de clusters aléatoirement
        self.initialize_centers(X)
        
        # Boucle principale de l'algorithme K-Means
        for iteration in range(self.max_iter):
            # Étape 2 : Assigner chaque point au cluster le plus proche
            # self.labels_ contient l'indice du cluster pour chaque point
            self.labels_ = self.nearest_cluster(X)
            
            # Sauvegarder les anciens centres pour vérifier la convergence
            old_centers = self.cluster_centers_.copy()
            
            # Étape 3 : Recalculer les centres de clusters (moyenne des points assignés)
            for i in range(self.n_clusters):
                # Trouver tous les points assignés au cluster i
                # self.labels_ == i crée un masque booléen
                # X[self.labels_ == i] sélectionne les points du cluster i
                points_in_cluster = X[self.labels_ == i]
                
                # Si le cluster n'est pas vide, calculer la moyenne
                if len(points_in_cluster) > 0:
                    self.cluster_centers_[i] = np.mean(points_in_cluster, axis=0)
            
            # Étape 4 : Vérifier la convergence (les centres ne bougent plus)
            # Si les centres n'ont pas changé, arrêter l'algorithme
            if np.allclose(old_centers, self.cluster_centers_):
                break

    def predict(self, X):
        """
        Prédit l'appartenance aux clusters pour les données X en utilisant les centres de clusters appris pendant l'entraînement.

        Entrée:
        - X (np.array): Les données d'entrée.

        Sortie:
        - np.array: Un tableau d'indices représentant le cluster prédit pour chaque point.
        """
        return self.nearest_cluster(X)




    

def show_metric(labels_true, labels_pred, descriptors,bool_return=False,name_descriptor="", name_model="kmeans",bool_show=True):
    """
    Fonction d'affichage et création des métrique pour le clustering.
    Input :
    - labels_true : étiquettes réelles des données
    - labels_pred : étiquettes prédites des données
    - descriptors : ensemble de descripteurs utilisé pour le clustering
    - bool_return : booléen indiquant si les métriques doivent être retournées ou affichées
    - name_descriptor : nom de l'ensemble de descripteurs utilisé pour le clustering
    - name_model : nom du modèle de clustering utilisé
    - bool_show : booléen indiquant si les métriques doivent être affichées ou non

    Output :
    - dictionnaire contenant les métriques d'évaluation des clusters
    """
    homogeneity, completeness, v_measure = metrics.homogeneity_completeness_v_measure(labels_true, labels_pred)
    jaccard = metrics.jaccard_score(labels_true, labels_pred, average='macro')
    ami = metrics.adjusted_mutual_info_score(labels_true, labels_pred)
    silhouette = silhouette_score(descriptors, labels_pred)
    ari = adjusted_rand_score(labels_true, labels_pred)
    # Affichons les résultats
    if bool_show :
        print(f"########## Métrique descripteur : {name_descriptor}")
        print(f"Adjusted Rand Index: {ari}")
        print(f"Jaccard Index: {jaccard}")
        print(f"Homogeneity: {homogeneity}")
        print(f"Completeness: {completeness}")
        print(f"V-measure: {v_measure}")
        print(f"Silhouette Score: {silhouette}")
        print(f"Adjusted Mutual Information: {ami}")
    if bool_return:
        return {"ami":ami,
                "ari":ari, 
                "silhouette":silhouette,
                "homogeneity":homogeneity,
                "completeness":completeness,
                "v_measure":v_measure, 
                "jaccard":jaccard,
               "descriptor":name_descriptor,
               "name_model":name_model}
