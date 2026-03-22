import os

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image

from constant import PATH_ALGO


@st.cache_data
def colorize_cluster(cluster_data, selected_cluster):
    fig = px.scatter_3d(cluster_data, x='x', y='y', z='z', color='cluster')
    filtered_data = cluster_data[cluster_data['cluster'] == selected_cluster]
    fig.add_scatter3d(x=filtered_data['x'], y=filtered_data['y'], z=filtered_data['z'],
                    mode='markers', marker=dict(color='red', size=10),
                    name=f'Cluster {selected_cluster}')
    return fig

@st.cache_data
def plot_metric(df_metric):
    # Création d'un histogramme pour le score AMI
    fig1 = px.bar(df_metric, x='descriptor', y='ami', 
                  title='Score AMI par descripteur',
                  labels={'ami': 'Score AMI', 'descriptor': 'Descripteur'},
                  color='descriptor')
    st.plotly_chart(fig1)


def load_excel_if_exists(file_path):
    if os.path.exists(file_path):
        return pd.read_excel(file_path)
    return None


# Sélection de l'algorithme (en haut de la sidebar)
st.sidebar.write("## Configuration du clustering")
algorithm = st.sidebar.selectbox('Algorithme de clustering', ["kmeans", "dbscan", "spectral", 'gmm'])

# Chargement dynamique des données selon l'algorithme
if algorithm == "kmeans":
    PATH_OUTPUT = os.path.join(PATH_ALGO, "kmeans_algo", "output")
elif algorithm == "dbscan":
    PATH_OUTPUT = os.path.join(PATH_ALGO, "dbscan_algo", "output")
elif algorithm == "spectral":
    PATH_OUTPUT = os.path.join(PATH_ALGO, "spectral_clustering_algo", "output")
elif algorithm == "gmm":
    PATH_OUTPUT = os.path.join(PATH_ALGO, "GMM_algo", "output")
else:
    st.error("Algorithme non reconnu.")
    st.stop()

df_hist = load_excel_if_exists(os.path.join(PATH_OUTPUT, f"save_clustering_hist_{algorithm}.xlsx"))
df_hog = load_excel_if_exists(os.path.join(PATH_OUTPUT, f"save_clustering_hog_{algorithm}.xlsx"))
df_resnet = load_excel_if_exists(os.path.join(PATH_OUTPUT, f"save_clustering_resnet_{algorithm}.xlsx"))
df_clip = load_excel_if_exists(os.path.join(PATH_OUTPUT, f"save_clustering_clip_{algorithm}.xlsx"))
df_vit = load_excel_if_exists(os.path.join(PATH_OUTPUT, f"save_clustering_vit_{algorithm}.xlsx"))
df_metric = load_excel_if_exists(os.path.join(PATH_OUTPUT, "save_metric.xlsx"))

if df_metric is None:
    st.error(f"Fichiers {algorithm.upper()} introuvables. Relance `pipeline_{algorithm}.py` pour générer les exports.")
    st.stop()

if 'Unnamed: 0' in df_metric.columns:
    df_metric.drop(columns="Unnamed: 0", inplace=True)

# Création de deux onglets
tab1, tab2 = st.tabs(["Analyse par descripteur", "Analyse global" ])

# Onglet numéro 1
with tab1:

    st.write(f'## Résultat de Clustering - {algorithm.upper()}')
    st.sidebar.write("####  Veuillez sélectionner les clusters à analyser" )
    # Sélection des descripteurs
    available_descriptors = []
    if df_hist is not None:
        available_descriptors.append("HISTOGRAM")
    if df_hog is not None:
        available_descriptors.append("HOG")
    if df_clip is not None:
        available_descriptors.append("CLIP")
    if df_resnet is not None:
        available_descriptors.append("RESNET")
    if df_vit is not None:
        available_descriptors.append("VIT")


    if len(available_descriptors) == 0:
        st.error(f"Aucun fichier clustering {algorithm.upper()} trouvé. Relance `pipeline_{algorithm}.py`.")
        st.stop()

    descriptor = st.sidebar.selectbox('Sélectionner un descripteur', available_descriptors)
    if descriptor=="HISTOGRAM":
        df = df_hist
    if descriptor=="HOG":
        df = df_hog
    if descriptor=="CLIP":
        df = df_clip
    if descriptor=="RESNET":
        df = df_resnet
    if descriptor=="VIT":
        df = df_vit

    # Nb de clusters
    unique_clusters = sorted(df['cluster'].unique())
    # Filtrer les clusters de bruit (-1) si DBSCAN
    if algorithm == "dbscan" and -1 in unique_clusters:
        cluster_options = [c for c in unique_clusters if c != -1]
        show_noise = st.sidebar.checkbox("Afficher les points de bruit (-1)", value=False)
        if show_noise:
            cluster_options = unique_clusters
    else:
        cluster_options = unique_clusters
    
    if len(cluster_options) == 0:
        st.warning("Aucun cluster valide trouvé.")
        st.stop()
    
    selected_cluster = st.sidebar.selectbox('Sélectionner un Cluster', cluster_options)
    # Filtrer les données en fonction du cluster sélectionné
    cluster_indices = df[df.cluster==selected_cluster].index    
    st.write(f"###  Analyse du descripteur {descriptor}" )
    st.write(f"#### Analyse du cluster : {selected_cluster} ({'Bruit' if selected_cluster == -1 else 'Cluster valide'})")
    st.write(f"####  Visualisation 3D - {algorithm.upper()} avec {descriptor}" )
    # Sélection du cluster choisi
    filtered_data = df[df['cluster'] == selected_cluster]
    # Création d'un graph 3D des clusters
    fig = colorize_cluster(df, selected_cluster)
    st.plotly_chart(fig)

    st.write(f"#### Images du cluster {selected_cluster}")
    if 'image_path' in filtered_data.columns:
        image_paths = filtered_data['image_path'].dropna().tolist()
        if len(image_paths) == 0:
            st.info("Aucune image trouvée pour ce cluster.")
        else:
            num_cols = 5
            cols = st.columns(num_cols)
            for i, img_path in enumerate(image_paths):
                col = cols[i % num_cols]
                img_path = os.path.normpath(str(img_path))
                if os.path.exists(img_path):
                    with col:
                        img = Image.open(img_path)
                        st.image(img, caption=os.path.basename(img_path), use_column_width=True)
                else:
                    with col:
                        st.warning(f"Image introuvable: {os.path.basename(img_path)}")
    else:
        st.warning("La colonne 'image_path' est absente. Relance `pipeline.py` pour régénérer les fichiers exportés.")

# Onglet numéro 2
with tab2:
    st.write('## Analyse Global des descripteurs' )
    # Complèter la fonction plot_metric() pour afficher les histogrammes du score AMI
    plot_metric(df_metric)
    st.write('## Métriques ' )
    # Affichage du tableau des métriques
    st.dataframe(df_metric)
