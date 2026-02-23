import os

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image

from constant import PATH_OUTPUT


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

        
# Chargement des données du clustering
df_hist = load_excel_if_exists(os.path.join(PATH_OUTPUT, "save_clustering_hist_kmeans.xlsx"))
df_hog = load_excel_if_exists(os.path.join(PATH_OUTPUT, "save_clustering_hog_kmeans.xlsx"))
df_clip = load_excel_if_exists(os.path.join(PATH_OUTPUT, "save_clustering_clip_kmeans.xlsx"))
df_metric = load_excel_if_exists(os.path.join(PATH_OUTPUT, "save_metric.xlsx"))

if df_metric is None:
    st.error("Fichier métriques introuvable. Relance `pipeline.py` pour générer les exports.")
    st.stop()

if 'Unnamed: 0' in df_metric.columns:
    df_metric.drop(columns="Unnamed: 0", inplace=True)

# Création de deux onglets
tab1, tab2 = st.tabs(["Analyse par descripteur", "Analyse global" ])

# Onglet numéro 1
with tab1:

    st.write('## Résultat de Clustering des données de snacks')
    st.sidebar.write("####  Veuillez sélectionner les clusters à analyser" )
    # Sélection des descripteurs
    available_descriptors = []
    if df_hist is not None:
        available_descriptors.append("HISTOGRAM")
    if df_hog is not None:
        available_descriptors.append("HOG")
    if df_clip is not None:
        available_descriptors.append("CLIP")

    if len(available_descriptors) == 0:
        st.error("Aucun fichier clustering trouvé. Relance `pipeline.py`.")
        st.stop()

    descriptor = st.sidebar.selectbox('Sélectionner un descripteur', available_descriptors)
    if descriptor=="HISTOGRAM":
        df = df_hist
    if descriptor=="HOG":
        df = df_hog
    if descriptor=="CLIP":
        df = df_clip

    # Nb de clusters
    num_clusters = df['cluster'].max() + 1
    selected_cluster =  st.sidebar.selectbox('Sélectionner un Cluster', range(int(num_clusters)))
    # Filtrer les données en fonction du cluster sélectionné
    cluster_indices = df[df.cluster==selected_cluster].index    
    st.write(f"###  Analyse du descripteur {descriptor}" )
    st.write(f"#### Analyse du cluster : {selected_cluster}")
    st.write(f"####  Visualisation 3D du clustering avec descripteur {descriptor}" )
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
