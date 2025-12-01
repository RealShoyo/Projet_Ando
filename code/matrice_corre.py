import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


OUTPUT_FILE = "matrice_correlation.png"
# 1. Charger le jeu de données
try:
    df = pd.read_csv("raw/final_dataset.csv")
except FileNotFoundError:
    print("Erreur: Le fichier 'final_dataset.csv' est introuvable. Veuillez vérifier le nom et l'emplacement du fichier.")
    exit()

# 2. Sélectionner uniquement les colonnes numériques
# Les matrices de corrélation ne peuvent être calculées qu'entre des variables numériques.
df_numeric = df.select_dtypes(include=[np.number])

# 3. Calculer la matrice de corrélation
# On utilise la méthode .corr() de pandas
correlation_matrix = df_numeric.corr()

# 4. Afficher le résultat (optionnel: vous pouvez imprimer la matrice brute)
print("--- Matrice de Corrélation de Pearson (Valeurs Brutes) ---")
print(correlation_matrix)
print("\n" + "="*70 + "\n")

# 5. Visualiser la matrice de corrélation sous forme de heatmap (carte de chaleur)
plt.figure(figsize=(16, 12)) # Ajuster la taille pour une meilleure lisibilité
sns.heatmap(
    correlation_matrix,
    annot=True,              # Afficher les valeurs de corrélation sur la carte
    fmt=".2f",               # Formater les nombres à deux décimales
    cmap='coolwarm',         # Palette de couleurs (coolwarm est idéale pour la corrélation)
    linewidths=.5,           # Lignes de séparation entre les cellules
    linecolor='black',
    cbar=True                # Afficher la barre de couleur
)

plt.title('Matrice de Corrélation des Statistiques de Pokémon', fontsize=18)


plt.tight_layout() # Ajuste la mise en page pour que les étiquettes ne soient pas coupées
plt.savefig(OUTPUT_FILE)

# --- Interprétation ---
print("--- Interprétation de la Heatmap ---")
print("* Les valeurs proches de **+1** (couleurs chaudes/rouges) indiquent une **corrélation positive forte**.")
print("  (Ex: si une stat augmente, l'autre augmente aussi.)")
print("* Les valeurs proches de **-1** (couleurs froides/bleues) indiquent une **corrélation négative forte**.")
print("  (Ex: si une stat augmente, l'autre diminue.)")
print("* Les valeurs proches de **0** (couleurs neutres/blanches) indiquent **aucune corrélation** linéaire.")