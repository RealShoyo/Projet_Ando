import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Chargement des données
df = pd.read_csv("raw/final_dataset.csv")

# Renommer la colonne 'Tiers' pour un accès plus facile si elle contient des espaces
df.columns = df.columns.str.replace(' ', '_')

# 2. Préparation des données
# Compter le nombre d'occurrences pour chaque valeur unique dans la colonne 'Tiers'
tier_counts = df['Tiers'].value_counts()

# Filtrer les Tiers que l'on ne veut pas afficher dans le graphique
# On enlève généralement 'Uber', 'NFE' (Not Fully Evolved), 'LC' (Little Cup), 'Unreleased', et les valeurs nulles
tiers_to_exclude = ['NFE', 'LC', 'Uber', 'Unreleased', 'illegal']
tier_counts_filtered = tier_counts[~tier_counts.index.isin(tiers_to_exclude)].sort_values(ascending=False)

# Si la colonne 'Tiers' contient des valeurs NaN, nous allons les exclure également.
if 'NULL' in tier_counts_filtered.index:
    tier_counts_filtered = tier_counts_filtered.drop('NULL')

# 3. Création du Diagramme à Barres
plt.figure(figsize=(12, 7))

# Utiliser seaborn pour un graphique plus esthétique, avec les tiers sur l'axe des x
sns.barplot(x=tier_counts_filtered.index, y=tier_counts_filtered.values, palette="viridis")

# Ajouter un titre et des étiquettes
plt.title('Distribution du Nombre de Pokémon par Tier de Compétitivité', fontsize=16)
plt.xlabel('Tier de Compétitivité', fontsize=12)
plt.ylabel('Nombre de Pokémon', fontsize=12)
plt.xticks(rotation=45, ha='right') # Rotation des étiquettes pour la lisibilité

# Afficher les valeurs au-dessus des barres
for index, value in enumerate(tier_counts_filtered.values):
    plt.text(index, value + 5, str(value), ha='center', va='bottom', fontsize=10)

plt.tight_layout() # Ajuster l'affichage pour éviter la coupure des étiquettes
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Afficher le graphique
plt.show()

print("\n--- Analyse des Tiers ---")
print("Répartition du nombre de Pokémon par Tier (hors NFE, LC, Uber, etc.):")
print(tier_counts_filtered)