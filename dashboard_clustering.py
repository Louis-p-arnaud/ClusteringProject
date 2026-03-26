import os
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image

import argparse

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_ANALYSIS_PATH = PROJECT_DIR / "outputs"
if not DEFAULT_ANALYSIS_PATH.exists():
    DEFAULT_ANALYSIS_PATH = PROJECT_DIR / "Algos"

parser = argparse.ArgumentParser()
parser.add_argument('-path_data', '--path_data', type=str, default=str(DEFAULT_ANALYSIS_PATH))
args, _ = parser.parse_known_args()

PATH_ALGO = os.path.abspath(args.path_data)

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


@st.cache_data
def plot_metric_grouped(df_metric):
    has_silhouette = "silhouette" in df_metric.columns
    has_dbi = "dbi" in df_metric.columns

    if not has_silhouette and not has_dbi:
        st.info("Aucune des métriques Silhouette/DBI n'est disponible dans le fichier métriques.")
        return

    col1, col2 = st.columns(2)

    with col1:
        if has_silhouette:
            fig_sil = px.bar(
                df_metric,
                x="descriptor",
                y="silhouette",
                color="descriptor",
                title="Silhouette par descripteur",
                labels={"descriptor": "Descripteur", "silhouette": "Silhouette"},
            )
            fig_sil.update_layout(
                height=420,
                margin=dict(l=40, r=20, t=80, b=90),
                showlegend=False,
            )
            fig_sil.update_xaxes(tickangle=-35, automargin=True)
            st.plotly_chart(fig_sil, use_container_width=True)
        else:
            st.info("Silhouette non disponible.")

    with col2:
        if has_dbi:
            fig_dbi = px.bar(
                df_metric,
                x="descriptor",
                y="dbi",
                color="descriptor",
                title="DBI par descripteur",
                labels={"descriptor": "Descripteur", "dbi": "DBI"},
            )
            fig_dbi.update_layout(
                height=420,
                margin=dict(l=40, r=20, t=80, b=90),
                showlegend=False,
            )
            fig_dbi.update_xaxes(tickangle=-35, automargin=True)
            st.plotly_chart(fig_dbi, use_container_width=True)
        else:
            st.info("DBI non disponible.")


def load_excel_if_exists(file_path):
    if os.path.exists(file_path):
        return pd.read_excel(file_path)
    return None


def load_algo_output_file(path_output, path_root, basename, algorithm):
    candidates = [
        os.path.join(path_output, f"{basename}_{algorithm}.xlsx"),
        os.path.join(path_root, f"{basename}_{algorithm}.xlsx"),
    ]
    for candidate in candidates:
        df = load_excel_if_exists(candidate)
        if df is not None:
            return df
    return None


def load_metric_file(path_output, path_root, algorithm):
    candidates = [
        os.path.join(path_output, "save_metric.xlsx"),
        os.path.join(path_root, f"save_metric_{algorithm}.xlsx"),
    ]
    for candidate in candidates:
        df = load_excel_if_exists(candidate)
        if df is not None:
            return df
    return None


def load_recap_metrics(path_algo):
    recap_rows = []
    algo_to_output = {
        "kmeans": "kmeans_algo",
        "dbscan": "dbscan_algo",
        "spectral": "spectral_clustering_algo",
        "gmm": "GMM_algo",
    }

    for model_name, algo_folder in algo_to_output.items():
        metric_candidates = [
            os.path.join(path_algo, algo_folder, "output", "save_metric.xlsx"),
            os.path.join(path_algo, f"save_metric_{model_name}.xlsx"),
        ]

        df_model = None
        for metric_path in metric_candidates:
            df_model = load_excel_if_exists(metric_path)
            if df_model is not None:
                break

        if df_model is None or len(df_model) == 0:
            continue

        if 'Unnamed: 0' in df_model.columns:
            df_model = df_model.drop(columns='Unnamed: 0')

        if 'descriptor' not in df_model.columns:
            continue

        for _, row in df_model.iterrows():
            recap_rows.append(
                {
                    "Modele": model_name.upper(),
                    "Descripteur": row.get("descriptor", "N/A"),
                    "Silhouette": row.get("silhouette", np.nan),
                    "DBI": row.get("dbi", np.nan),
                    "AMI": row.get("ami", np.nan),
                }
            )

    if len(recap_rows) == 0:
        return pd.DataFrame(columns=["Modele", "Descripteur", "Silhouette", "DBI", "AMI"])

    df_recap = pd.DataFrame(recap_rows)
    return df_recap.sort_values(by=["Modele", "Descripteur"]).reset_index(drop=True)


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

df_hist = load_algo_output_file(PATH_OUTPUT, PATH_ALGO, "save_clustering_hist", algorithm)
df_hog = load_algo_output_file(PATH_OUTPUT, PATH_ALGO, "save_clustering_hog", algorithm)
df_resnet = load_algo_output_file(PATH_OUTPUT, PATH_ALGO, "save_clustering_resnet", algorithm)
df_clip = load_algo_output_file(PATH_OUTPUT, PATH_ALGO, "save_clustering_clip", algorithm)
df_vit = load_algo_output_file(PATH_OUTPUT, PATH_ALGO, "save_clustering_vit", algorithm)
df_metric = load_metric_file(PATH_OUTPUT, PATH_ALGO, algorithm)
df_silhouette_curve = None
if algorithm == "kmeans":
    df_silhouette_curve = load_excel_if_exists(os.path.join(PATH_OUTPUT, "save_silhouette_curve_kmeans.xlsx"))
    if df_silhouette_curve is None:
        df_silhouette_curve = load_excel_if_exists(os.path.join(PATH_ALGO, "save_silhouette_curve_kmeans.xlsx"))

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

                img_path_str = str(img_path)
                candidate_paths = [os.path.normpath(img_path_str)]
                if not os.path.isabs(img_path_str):
                    candidate_paths.append(os.path.normpath(os.path.join(str(PROJECT_DIR), img_path_str)))

                existing_path = None
                for candidate in candidate_paths:
                    if os.path.exists(candidate):
                        existing_path = candidate
                        break

                if existing_path is not None:
                    with col:
                        img = Image.open(existing_path)
                        st.image(img, caption=os.path.basename(existing_path), use_column_width=True)
                else:
                    with col:
                        st.warning(f"Image introuvable: {os.path.basename(img_path_str)}")
    else:
        st.warning("La colonne 'image_path' est absente. Relance `pipeline.py` pour régénérer les fichiers exportés.")

# Onglet numéro 2
with tab2:
    st.write('## Analyse Global des descripteurs' )
    # Complèter la fonction plot_metric() pour afficher les histogrammes du score AMI
    plot_metric(df_metric)
    st.write('## Comparaison Silhouette / DBI')
    plot_metric_grouped(df_metric)

    if algorithm == "kmeans":
        st.write('## Evolution du Silhouette Score (KMeans)')
        if df_silhouette_curve is not None and len(df_silhouette_curve) > 0:
            fig_silhouette = px.line(
                df_silhouette_curve,
                x='k',
                y='silhouette',
                color='descriptor',
                markers=True,
                title='Silhouette Score selon K (5, 10, 15, 20, 25)',
                labels={'k': 'Nombre de clusters K', 'silhouette': 'Silhouette Score', 'descriptor': 'Descripteur'}
            )
            st.plotly_chart(fig_silhouette)
        else:
            st.info("Courbe de silhouette non trouvée. Relance `pipeline_kmeans.py` pour générer `save_silhouette_curve_kmeans.xlsx`.")

    st.write('## Métriques ' )
    # Affichage du tableau des métriques
    st.dataframe(df_metric)

    st.write('## Recap')
    df_recap = load_recap_metrics(PATH_ALGO)
    if len(df_recap) == 0:
        st.info("Aucune métrique trouvée pour le récapitulatif global. Relance les pipelines pour générer les fichiers `save_metric.xlsx`.")
    else:
        st.dataframe(df_recap)
