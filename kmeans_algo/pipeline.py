from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
from sklearn import datasets
import numpy as np

from features import *
from clustering import *
from utils import *
from constant import PATH_OUTPUT, MODEL_CLUSTERING



def pipeline():
   
    digits = datasets.load_digits()


    labels_true =digits.target
    images = digits.images
   
    print("\n\n ##### Extraction de Features ######")
    print("- calcul features hog...")
    descriptors_hog = compute_hog_descriptors(images)
    print("- calcul features Histogram...")
    descriptors_hist = compute_gray_histograms(images)


    print("\n\n ##### Clustering ######")
    number_cluster = 10
    kmeans_hog = KMeans(n_clusters=number_cluster)
    kmeans_hist = KMeans(n_clusters=number_cluster)
    
    print("- calcul kmeans avec features HOG ...")
    kmeans_hog.fit(np.array(descriptors_hog))
    print("- calcul kmeans avec features Histogram...")
    kmeans_hist.fit(np.array(descriptors_hist))


    print("\n\n ##### Résultat ######")
    metric_hist = show_metric(labels_true, kmeans_hist.labels_, descriptors_hist, bool_show=True, name_descriptor="HISTOGRAM", bool_return=True)
    metric_hog = show_metric(labels_true, kmeans_hog.labels_, descriptors_hog,bool_show=True, name_descriptor="HOG", bool_return=True)


    print("- export des données vers le dashboard")
    # conversion des données vers le format du dashboard
    list_dict = [metric_hist,metric_hog]
    df_metric = pd.DataFrame(list_dict)
    
    # Normalisation des données
    scaler = StandardScaler()
    descriptors_hist_norm = scaler.fit_transform(descriptors_hist)
    descriptors_hog_norm = scaler.fit_transform(descriptors_hog)

    #conversion vers un format 3D pour la visualisation
    x_3d_hist = conversion_3d(descriptors_hist_norm)
    x_3d_hog = conversion_3d(descriptors_hog_norm)

    # création des dataframe pour la sauvegarde des données pour la visualisation
    df_hist = create_df_to_export(x_3d_hist, labels_true, kmeans_hist.labels_)
    df_hog = create_df_to_export(x_3d_hog, labels_true, kmeans_hog.labels_)

    # Vérifie si le dossier existe déjà
    if not os.path.exists(PATH_OUTPUT):
        # Crée le dossier
        os.makedirs(PATH_OUTPUT)

    # sauvegarde des données
    df_hist.to_excel(PATH_OUTPUT+"/save_clustering_hist_kmeans.xlsx")
    df_hog.to_excel(PATH_OUTPUT+"/save_clustering_hog_kmeans.xlsx")
    df_metric.to_excel(PATH_OUTPUT+"/save_metric.xlsx")
    print("Fin. \n\n Pour avoir la visualisation dashboard, veuillez lancer la commande : streamlit run dashboard_clustering.py")


if __name__ == "__main__":
    pipeline()