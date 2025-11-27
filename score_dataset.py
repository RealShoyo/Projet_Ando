from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql.functions import col, rank, max as spark_max, regexp_replace, when, lit, row_number
spark = SparkSession.builder \
    .appName("CSVReaderApp") \
    .getOrCreate()

file_path = "usage_by_format.csv"

df_usage = spark.read.csv(
    path=file_path, 
    header=True, 
    inferSchema=True, 
    sep=','
)


df_clean = df_usage.withColumn(
    "Percent_Float",
    regexp_replace(col("Usage %"), "%", "").cast("float")
).drop("Percent")

# Définition de la fenêtre : partition par 'tiers' et classement par 'Percent_Float' décroissant
window_spec = Window.partitionBy("Tiers").orderBy(col("Percent_Float").desc())

# Attribution du rang (Rank) à chaque Pokémon au sein de son 'tiers'
df_ranked = df_clean.withColumn("rank", rank().over(window_spec))

# Filtrer uniquement les Pokémon ayant le rang 1 (le plus haut Percent)
df_top_by_tier = df_ranked.filter(col("rank") == 1).drop("rank")

# Renommer la colonne pour la clarté finale
df_final_spark = df_top_by_tier.withColumnRenamed("Percent_Float", "Max_Percent_Usage")

df_joined = df_clean.join(df_final_spark.select("tiers", "Max_Percent_Usage"), on="tiers", how="left")

df_scored = df_joined.withColumn(
    "Tier_Score",
    when(col("tiers") == "pu", 0 + col("Percent_Float") / col("Max_Percent_Usage"))
    .when(col("tiers") == "nu", 1 + col("Percent_Float") / col("Max_Percent_Usage"))
    .when(col("tiers") == "ru", 2 + col("Percent_Float") / col("Max_Percent_Usage"))
    .when(col("tiers") == "uu", 3 + col("Percent_Float") / col("Max_Percent_Usage"))
    .when(col("tiers") == "ou", 4 + col("Percent_Float") / col("Max_Percent_Usage"))
    # Gère tous les autres tiers non spécifiés (ex: AG, LC, NFE)
    .otherwise(lit(None)) 
).drop("Max_Percent_Usage")

window_spec_final = Window.partitionBy("Pokemon").orderBy(col("Tier_Score").asc())

# Attribuer un numéro de ligne à chaque entrée de Pokémon
# Le numéro 1 sera le Pokémon avec le Tier_Score le plus petit
df_final_ranked = df_scored.withColumn("row_num", row_number().over(window_spec_final))

# Filtrer pour ne garder que le rang 1 (le plus petit Tier_Score)
df_unique_min_tier = df_final_ranked.filter(col("row_num") == 1).drop("row_num")

# ----------------------------------------------------
# --- ÉTAPE D'ÉCRITURE DU FICHIER ---
# ----------------------------------------------------

output_path = "scored_usage_data.csv"

# 1. Sélectionner les colonnes finales et l'ordre souhaité
df_final_output = df_unique_min_tier.select(
    "Pokemon", 
    "Tiers", 
    col("Percent_Float").alias("Percent_Usage"), # Renommer pour la clarté
    "Tier_Score"
)

df_pandas = df_final_output.toPandas()

df_pandas.to_csv(output_path, index=False)

# Arrêt de la session Spark
spark.stop()