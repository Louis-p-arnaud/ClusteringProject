from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import numpy as np
from sklearn import metrics


class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        """
        Initialise un objet DBSCAN.

        Entrées:
        - eps (float): Distance maximale entre deux points pour être considérés voisins (par défaut 0.5).
        - min_samples (int): Nombre minimum de points pour former un cluster dense (par défaut 5).
        """
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def _region_query(self, X, point_idx):
        """
        Trouve tous les points voisins dans un rayon eps autour d'un point donné.

        Entrée:
        - X (np.array): Les données d'entrée.
        - point_idx (int): L'indice du point central.

        Sortie:
        - np.array: Indices des points voisins.
        """
        distances = np.sqrt(np.sum((X - X[point_idx]) ** 2, axis=1))
        return np.where(distances <= self.eps)[0]

    def _expand_cluster(self, X, labels, point_idx, neighbors, cluster_id):
        """
        Expand un cluster à partir d'un point core en ajoutant récursivement ses voisins.

        Entrée:
        - X (np.array): Les données d'entrée.
        - labels (np.array): Tableau des labels actuels.
        - point_idx (int): Indice du point core.
        - neighbors (np.array): Indices des voisins initiaux.
        - cluster_id (int): ID du cluster en cours de construction.

        Sortie:
        - bool: True si le cluster a été créé.
        """
        labels[point_idx] = cluster_id
        
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            
            # Si le point est du bruit (-1), le reclasser dans ce cluster
            if labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id
            
            # Si le point n'a pas encore été visité (label 0)
            elif labels[neighbor_idx] == 0:
                labels[neighbor_idx] = cluster_id
                
                # Chercher les voisins de ce point
                neighbor_neighbors = self._region_query(X, neighbor_idx)
                
                # Si c'est un point core, ajouter ses voisins à la liste
                if len(neighbor_neighbors) >= self.min_samples:
                    neighbors = np.concatenate([neighbors, neighbor_neighbors])
            
            i += 1
        
        return True

    def fit(self, X):
        """
        Exécute l'algorithme DBSCAN sur les données X.

        Entrée:
        - X (np.array): Les données d'entrée.

        Sortie:
        - Aucune sortie directe, mais les labels sont stockés dans self.labels_.
        """
        n_points = X.shape[0]
        labels = np.zeros(n_points, dtype=int)  # 0 = non visité
        cluster_id = 0
        
        for point_idx in range(n_points):
            # Si le point a déjà été visité, passer au suivant
            if labels[point_idx] != 0:
                continue
            
            # Trouver les voisins du point
            neighbors = self._region_query(X, point_idx)
            
            # Si pas assez de voisins, c'est du bruit
            if len(neighbors) < self.min_samples:
                labels[point_idx] = -1  # -1 = bruit
            else:
                # Point core : créer un nouveau cluster
                cluster_id += 1
                self._expand_cluster(X, labels, point_idx, neighbors, cluster_id)
        
        self.labels_ = labels

    def predict(self, X):
        """
        DBSCAN n'a pas de méthode predict standard (pas de centres de clusters).
        Cette méthode retourne les labels appris pendant fit.
        
        Entrée:
        - X (np.array): Les données d'entrée (ignorées).

        Sortie:
        - np.array: Labels du clustering.
        """
        if self.labels_ is None:
            raise ValueError("Le modèle doit être entraîné avec fit() avant de prédire.")
        return self.labels_


def show_metric(labels_true, labels_pred, descriptors, bool_return=False, name_descriptor="", name_model="dbscan", bool_show=True):
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
    # Filtrer les points de bruit (-1) pour certaines métriques
    mask = labels_pred != -1
    labels_pred_filtered = labels_pred[mask]
    labels_true_filtered = labels_true[mask]
    descriptors_filtered = descriptors[mask]
    
    # Vérifier qu'il reste au moins 2 clusters
    n_clusters = len(set(labels_pred_filtered)) - (1 if -1 in labels_pred_filtered else 0)
    
    if n_clusters < 2 or len(labels_pred_filtered) < 2:
        print(f"########## Métrique descripteur : {name_descriptor} - ÉCHEC (moins de 2 clusters)")
        if bool_return:
            return {
                "ami": 0.0, "ari": 0.0, "silhouette": 0.0,
                "homogeneity": 0.0, "completeness": 0.0, "v_measure": 0.0,
                "jaccard": 0.0, "descriptor": name_descriptor, "name_model": name_model
            }
        return None
    
    homogeneity, completeness, v_measure = metrics.homogeneity_completeness_v_measure(labels_true, labels_pred)
    jaccard = metrics.jaccard_score(labels_true, labels_pred, average='macro')
    ami = metrics.adjusted_mutual_info_score(labels_true, labels_pred)
    
    # Silhouette seulement sur les points non-bruit
    silhouette = silhouette_score(descriptors_filtered, labels_pred_filtered) if len(labels_pred_filtered) > 1 else 0.0
    ari = adjusted_rand_score(labels_true, labels_pred)
    
    # Affichons les résultats
    if bool_show:
        print(f"########## Métrique descripteur : {name_descriptor}")
        print(f"Adjusted Rand Index: {ari}")
        print(f"Jaccard Index: {jaccard}")
        print(f"Homogeneity: {homogeneity}")
        print(f"Completeness: {completeness}")
        print(f"V-measure: {v_measure}")
        print(f"Silhouette Score: {silhouette}")
        print(f"Adjusted Mutual Information: {ami}")
        print(f"Nombre de clusters trouvés: {n_clusters}")
        print(f"Points de bruit: {np.sum(labels_pred == -1)}")
    
    if bool_return:
        return {
            "ami": ami,
            "ari": ari,
            "silhouette": silhouette,
            "homogeneity": homogeneity,
            "completeness": completeness,
            "v_measure": v_measure,
            "jaccard": jaccard,
            "descriptor": name_descriptor,
            "name_model": name_model
        }
