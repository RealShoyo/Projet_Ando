from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

output_path = "final_dataset.csv"

spark = SparkSession.builder \
    .appName("PokemonLeftJoin") \
    .getOrCreate()

# Chemins des fichiers
dataset_file = "dataset_v2.csv"
scored_usage_file = "scored_usage_data.csv"


df_base = spark.read.csv(
    path=dataset_file,
    header=True,
    inferSchema=True
)

df_scores = spark.read.csv(
    path=scored_usage_file,
    header=True,
    inferSchema=True
)


df_base = df_base.withColumnRenamed("Pokemon Name", "Pokemon")

df_joined = df_base.join(
    other=df_scores, 
    on="Pokemon", 
    how="left"
)



# Affichage du schéma et d'un aperçu
print("Schéma du DataFrame après LEFT JOIN :")
df_joined.printSchema()

print("\nAperçu du DataFrame joint (avec les colonnes du score) :")
# On affiche les colonnes du dataset de base + les nouvelles colonnes jointes
df_joined.select(
    "Pokemon", 
    "Health Stat", 
    "Primary Type", 
    "Tiers", 
    "Tier_Score"
).show(10, truncate=False)

# Si certains Pokémon n'ont pas de score, les colonnes 'tiers' et 'Tier_Score' seront NULL.

df_final = df_joined.select(
    df_base["*"], # Conserve TOUTES les colonnes du DataFrame de base
    df_scores["Tiers"],
    df_scores["Percent_Usage"],
    df_scores["Tier_Score"]
)

print("\nSchéma du DataFrame FINAL après sélection :")
df_final.where(col("Tier_Score").isNotNull()).select("Pokemon", "Tier_Score").show()
df_final = df_final.where(col("Tier_Score").isNotNull()).dropDuplicates().orderBy("Pokedex Number")
df_final = df_final.drop("Pokedex Number", "Primary Type", "Secondary Type", "Male Ratio", "Female Ratio", "Base Happiness", "Game(s) of Origin", "Health EV", "Attack EV", "Defense EV", "Special Attack EV", "Special Defense EV", "Speed EV", "EV Yield Total", "Experience Growth Total")
df_final = df_final.withColumn("Experience Growth", 
                               when(col("Experience Growth") == "Fast", 0)
                               .when(col("Experience Growth") == "Medium Fast", 1)
                               .when(col("Experience Growth") == "Medium Slow", 2)
                               .when(col("Experience Growth") == "Slow", 3))
df_pandas = df_final.toPandas()

df_pandas.to_csv(output_path, index=False)

# Arrêt de la session Spark
spark.stop()