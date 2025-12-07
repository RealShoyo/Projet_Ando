import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 

# Charger le fichier CSV
# NOTE: Le chemin est ajusté pour utiliser le fichier téléversé "final_dataset.csv"
df = pd.read_csv("raw/final_dataset.csv") 
df.columns = df.columns.str.replace(' ', '_')

# Définition des groupes de colonnes et des k optimaux
analysis_config = {
    "Statistiques_de_Combat": {
        "cols": ['Health_Stat', 'Attack_Stat', 'Defense_Stat', 'Special_Attack_Stat', 'Special_Defense_Stat', 'Speed_Stat', 'Base_Stat_Total'],
        "k": 4
    },
    "Poids_et_Taille": {
        "cols": ['Pokemon_Height', 'Pokemon_Weight'],
        "k": 3
    },
    "Taux_de_Capture": {
        "cols": ['Catch_Rate'],
        "k": 4
    },
    "Cycles_d'Oeuf": {
        "cols": ['Egg_Cycle_Count'],
        "k": 5
    }
}

# Fonction de préparation des données (gestion des NaN et standardisation)
def prepare_data(df, columns):
    data = df[columns].copy()
    for col in columns:
        data[col] = data[col].fillna(data[col].median())
        
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    return scaled_data, scaler

# Fonction principale pour exécuter K-means avec et sans PCA
def run_final_clustering(group_name, config, df):
    k = config["k"]
    original_cols = config["cols"]
    scaled_data, scaler = prepare_data(df, original_cols)
    
    print(f"==================================================")
    print(f"## 🏆 Analyse : {group_name} (k={k} optimal)")
    print(f"==================================================")
    
    # --- A) K-means SANS PCA ---
    
    print("\n--- A) K-means SANS PCA (Analyse Principale) ---")
    
    kmeans_no_pca = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters_no_pca = kmeans_no_pca.fit_predict(scaled_data)
    
    df[f'Cluster_{group_name}_NoPCA'] = clusters_no_pca
    
    # Afficher les centres des clusters (interprétation)
    centers_no_pca = pd.DataFrame(kmeans_no_pca.cluster_centers_, columns=[f'Std_{col}' for col in original_cols])
    
    print(f"\nCentres des Clusters (Échelle Standardisée, K={k}):")
    print(centers_no_pca.round(2))
    
    # Analyse de la compétitivité (Tier_Score)
    tier_score_no_pca = df.groupby(f'Cluster_{group_name}_NoPCA')['Tier_Score'].agg(['count', 'mean', 'median', 'std'])
    print(f"\nTier_Score moyen par Cluster (SANS PCA) :")
    print(tier_score_no_pca.sort_values(by='mean', ascending=False).round(4))
    
    # --- C) Visualisation SANS PCA (pour 1 variable) ---
    
    if scaled_data.shape[1] == 1:
        # Visualisation 1D pour les groupes de variable unique (Catch Rate, Egg Cycle Count)
        print(f"\nVisualisation 1D des Clusters : {original_cols[0]} (SANS PCA)")
        
        plt.figure(figsize=(10, 3))
        # Utiliser un jitter sur l'axe des ordonnées pour mieux visualiser les points
        y_jitter = np.random.uniform(-0.1, 0.1, size=scaled_data.shape[0])
        plt.scatter(scaled_data[:, 0], y_jitter, 
                    c=clusters_no_pca, cmap='viridis', marker='o', alpha=0.6)
        
        # Afficher les centres des clusters (sur la ligne y=0)
        centers_no_pca_1d = kmeans_no_pca.cluster_centers_
        plt.scatter(centers_no_pca_1d, np.zeros_like(centers_no_pca_1d), 
                    marker='X', s=200, c='red', label='Centres de Cluster')
        
        plt.title(f'K-means sur {group_name} (k={k}) - SANS PCA')
        plt.xlabel(f'{original_cols[0]} (Standardisé)')
        plt.yticks([]) # Pas d'axe des ordonnées significatif
        plt.legend()
        plt.grid(axis='x')
        plt.show()
    
    
    # --- B) K-means AVEC PCA (pour les groupes > 1 variable) ---
    
    if len(original_cols) > 1:
        print("\n--- B) K-means AVEC PCA ---")
        
        # PCA avec 2 composantes pour la visualisation (si possible)
        n_components = min(scaled_data.shape[1], 2)
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(scaled_data)
        
        explained_variance_ratio = pca.explained_variance_ratio_
        explained_variance = explained_variance_ratio.sum() * 100
        print(f"Variance expliquée par les {n_components} premières composantes : {explained_variance:.1f}%")
        
        
        # ************************************************************
        # PARTIE NOUVELLE : INTERPRÉTATION DES COMPOSANTES PRINCIPALES
        # ************************************************************
        
        print("\n### 🔍 Interprétation des Composantes Principales (PC)")
        
        # Créer le DataFrame des coefficients (Loadings)
        pca_loadings = pd.DataFrame(
            pca.components_.T, # Transposé de components_ pour avoir les variables en index
            columns=[f'PC{i+1} ({explained_variance_ratio[i]*100:.1f}%)' for i in range(n_components)], 
            index=original_cols
        )
        print("Coefficients de Contribution des Variables (Loadings):")
        print(pca_loadings.round(3))
        
        # Ajout d'une analyse textuelle basée sur le nom du groupe
        if group_name == "Statistiques_de_Combat":
            print("\nAnalyse PC1: Représente la **Puissance Globale**. Elle est dominée par des coefficients positifs sur TOUTES les statistiques.")
            print("Analyse PC2: Représente le **Style de Combat**. Cherchez l'opposition (signes opposés) entre les stats Offensives et Défensives, ou entre Attaque et Vitesse.")
        elif group_name == "Poids_et_Taille":
            print("\nAnalyse PC1: Représente la **Masse/Volume Physique**. Elle est dominée par des coefficients positifs sur Poids et Taille.")
            
        # ************************************************************
        
        kmeans_pca = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters_pca = kmeans_pca.fit_predict(principal_components)
        
        df[f'Cluster_{group_name}_PCA'] = clusters_pca
        
        # Analyse de la compétitivité (Tier_Score)
        tier_score_pca = df.groupby(f'Cluster_{group_name}_PCA')['Tier_Score'].agg(['count', 'mean', 'median', 'std'])
        print(f"\nTier_Score moyen par Cluster (AVEC PCA) :")
        print(tier_score_pca.sort_values(by='mean', ascending=False).round(4))
        
        # Visualisation PCA (2D)
        if n_components >= 2:
            print("\nVisualisation des Clusters (basée sur les deux premières Composantes Principales) :")
            pca_df = pd.DataFrame(data = principal_components[:,:2], columns = ['PC1', 'PC2'])
            pca_df[f'Cluster'] = clusters_pca
            
            plt.figure(figsize=(10, 7))
            plt.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['Cluster'], cmap='viridis', marker='o', alpha=0.6)
            centers_pca = kmeans_pca.cluster_centers_
            plt.scatter(centers_pca[:, 0], centers_pca[:, 1], marker='X', s=200, c='red', label='Centres de Cluster')
            
            plt.title(f'K-means sur {group_name} (k={k}, après PCA)')
            plt.xlabel(f'Composante Principale 1 ({explained_variance_ratio[0]*100:.1f}%)')
            plt.ylabel(f'Composante Principale 2 ({explained_variance_ratio[1]*100:.1f}%)')
            plt.legend()
            plt.grid(True)
            plt.show()
    
    print("\n--------------------------------------------------\n")

# Exécuter l'analyse pour chaque groupe
for name, config in analysis_config.items():
    run_final_clustering(name, config, df)

# Correction du chemin de sauvegarde
print("Aperçu du DataFrame avec tous les résultats de clustering :")
print(df[['Pokemon', 'Tier_Score'] + [col for col in df.columns if 'Cluster' in col]].head())
df.to_csv("res_cluster.csv", index=False)