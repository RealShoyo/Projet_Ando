import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# 1. Chargement et Préparation des Données
df = pd.read_csv("raw/final_dataset.csv")
df.columns = df.columns.str.replace(' ', '_')

# Sélectionner le groupe de statistiques pour l'analyse
stat_cols = ['Health_Stat', 'Attack_Stat', 'Defense_Stat', 
             'Special_Attack_Stat', 'Special_Defense_Stat', 
             'Speed_Stat', 'Base_Stat_Total']

# Fonction de préparation des données (reprise de l'analyse précédente)
def prepare_data(df, columns):
    data = df[columns].copy()
    for col in columns:
        data[col] = data[col].fillna(data[col].median())
        
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    return scaled_data

# Préparer les données standardisées
scaled_data = prepare_data(df, stat_cols)
# Appliquer PCA pour la version avec réduction de dimension
pca = PCA(n_components=2) # On pourrait en prendre plus, mais 2 suffisent pour illustrer
principal_components = pca.fit_transform(scaled_data)

# Définir l'intervalle de k à tester
k_range = range(1, 11)

def run_elbow_method(data, title_suffix):
    """
    Calcule l'Inertia (WCSS) pour chaque valeur de k et affiche le graphique.
    """
    inertia_values = []
    
    # Calculer l'Inertia pour chaque k
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        inertia_values.append(kmeans.inertia_)
    
    # Affichage du graphique
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertia_values, marker='o', linestyle='-', color='b')
    plt.title(f"Méthode du Coude : {title_suffix}")
    plt.xlabel("Nombre de Clusters (k)")
    plt.ylabel("Inertie (WCSS)")
    plt.xticks(k_range)
    plt.grid(True)
    plt.show()
    
    print(f"Les valeurs d'Inertie pour {title_suffix} (k=1 à k=10) sont:")
    for k, inertia in zip(k_range, inertia_values):
        print(f"k={k}: {inertia:.2f}")

# 2. Exécuter la Méthode du Coude SANS PCA
print("==================================================")
print("## Méthode du Coude SANS PCA (sur toutes les stats)")
print("==================================================")
run_elbow_method(scaled_data, "Statistiques de Combat (SANS PCA)")

# 3. Exécuter la Méthode du Coude AVEC PCA
print("\n==================================================")
print("## Méthode du Coude AVEC PCA (sur 2 Composantes Principales)")
print("==================================================")
# Nous utilisons ici les 2 composantes principales pour simuler le cas d'une réduction de dimension
run_elbow_method(principal_components, "Statistiques de Combat (AVEC PCA, 2 Composantes)")

print("\n--- Analyse ---")
print("Veuillez observer les deux graphiques et chercher le point où la diminution de l'Inertie commence à ralentir de manière significative. C'est le 'coude'.")