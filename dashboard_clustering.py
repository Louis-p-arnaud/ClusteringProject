import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from constant import PATH_OUTPUT

# Configuration de la page
st.set_page_config(page_title="Dashboard Clustering Snacks", layout="wide")

@st.cache_data
def load_data(file_name):
    path = os.path.join(PATH_OUTPUT, file_name)
    if os.path.exists(path):
        return pd.read_excel(path)
    return None

def colorize_cluster(cluster_data, selected_cluster):
    # Création du graphe 3D [cite: 35]
    fig = px.scatter_3d(
        cluster_data, x='x', y='y', z='z', 
        color='cluster',
        title=f"Exploration 3D des Clusters",
        labels={'x': 'PCA 1', 'y': 'PCA 2', 'z': 'PCA 3'}
    )
    
    # Mise en évidence du cluster sélectionné
    filtered_data = cluster_data[cluster_data['cluster'] == selected_cluster]
    fig.add_scatter3d(
        x=filtered_data['x'], y=filtered_data['y'], z=filtered_data['z'],
        mode='markers', 
        marker=dict(color='red', size=8, symbol='diamond'),
        name=f'Focus Cluster {selected_cluster}'
    )
    return fig

# --- CHARGEMENT DES DONNÉES ---
df_hist = load_data("save_clustering_hist_kmeans.xlsx")
df_hog = load_data("save_clustering_hog_kmeans.xlsx")
df_vit = load_data("save_clustering_vit_spectral.xlsx") # Ton nouveau modèle [cite: 7]
df_metric = load_data("save_metric.xlsx")
df_silhouette = load_data("silhouette_scores.xlsx") # Pour le graphe de suivi [cite: 39]

# --- INTERFACE ---
st.title("Projet IA ET4 : Clustering de Snacks 🍩") 

tab1, tab2 = st.tabs(["Analyse par descripteur", "Analyse globale"]) 
with tab1:
    st.sidebar.header("Configuration de l'analyse")
    
    # Sélection du descripteur [cite: 8, 36]
    model_choice = st.sidebar.selectbox(
        'Sélectionner le modèle (Descripteur + Algo)', 
        ["HIST + K-Means", "HOG + K-Means", "ViT + Spectral Clustering"]
    )
    
    # Assignation du DataFrame en fonction du choix
    if "HIST" in model_choice:
        df = df_hist
    elif "HOG" in model_choice:
        df = df_hog
    else:
        df = df_vit

    if df is not None:
        # Sélection du cluster [cite: 36]
        num_clusters = int(df['cluster'].max() + 1)
        selected_cluster = st.sidebar.slider('Sélectionner un Cluster', 0, num_clusters - 1, 0)

        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.subheader(f"Visualisation 3D : {model_choice}") 
            fig_3d = colorize_cluster(df, selected_cluster)
            st.plotly_chart(fig_3d, use_container_width=True)

        with col_right:
            st.subheader("Métriques du modèle")
            if df_metric is not None:
                # Filtrer la métrique correspondant au descripteur choisi
                m_name = model_choice.split(" ")[0].lower()
                current_metric = df_metric[df_metric['descriptor'].str.contains(m_name, case=False)]
                if not current_metric.empty:
                    st.metric("Score AMI", f"{current_metric['ami'].values[0]:.4f}")
                    st.metric("Silhouette Score", f"{current_metric['silhouette'].values[0]:.4f}")
       
       

        st.divider()
        st.subheader(f"Images du cluster {selected_cluster}")

        if 'image_path' in df.columns:
            cluster_images = df[df['cluster'] == selected_cluster]['image_path'].dropna().tolist()
            
            if cluster_images:
                n_cols = 4
                rows = (len(cluster_images) // n_cols) + 1
                for i in range(rows):
                    cols = st.columns(n_cols)
                    for j in range(n_cols):
                        idx = i * n_cols + j
                        if idx < len(cluster_images):
                            # On récupère le nom du fichier et son dossier parent (la classe)
                            path_parts = cluster_images[idx].replace("\\", "/").split("/")
                            filename = path_parts[-1]
                            classname = path_parts[-2]
                            
                            # On reconstruit le chemin basé sur TON architecture : dossier 'dataset'
                            # Cela garantit que ça marche sur ton PC et dans Docker
                            local_path = os.path.join("dataset", classname, filename)

                            if os.path.exists(local_path):
                                cols[j].image(local_path, use_column_width=True)
                            else:
                                cols[j].caption(f"⚠️ {filename}")
                                # Optionnel : décommente la ligne suivante pour debugger le chemin
                                # cols[j].write(f"Tenté: {local_path}")
            else:
                st.warning("Aucune image dans ce cluster.")

with tab2:
    st.header("Comparaison Globale des Performances") 
    
    if df_metric is not None:
        st.subheader("Tableau comparatif des métriques")
        st.dataframe(df_metric, use_container_width=True)
        
        fig_ami = px.bar(df_metric, x='descriptor', y='ami', color='descriptor', title="Comparaison du score AMI")
        st.plotly_chart(fig_ami)

    if df_silhouette is not None:
        st.subheader("Graphe de suivi du Silhouette Score")
        # Affichage pour 5, 10, 15, 20, 25 clusters [cite: 40]
        fig_sil = px.line(
            df_silhouette, x='n_clusters', y='silhouette_score', 
            markers=True, title="Évolution de la qualité du clustering (K-Means)"
        )
        st.plotly_chart(fig_sil)