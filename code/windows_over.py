from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql.functions import col, rank, max as spark_max, regexp_replace, when, lit, row_number, concat

spark = SparkSession.builder.appName("Windows pour certains paramètres").getOrCreate()

dataset = "raw/final_dataset.csv"

df_data = spark.read.csv(
    path=dataset, 
    header=True, 
    inferSchema=True, 
    sep=','
)


window_spec = Window.partitionBy("Experience Growth").orderBy(col("Tier_Score").desc())
df_data = df_data.select("Experience Growth", "Tier_Score", "Primary Egg Group", "Secondary Egg Group")
# Experience Growth
df_value = df_data.groupBy("Experience Growth").avg("Tier_Score")
df_value.show()
#Egg Group
df_egg1 = df_data.select(col("Primary Egg Group").alias("Egg_Group"), col("Tier_Score"))
df_egg2 = df_data.select(col("Secondary Egg Group").alias("Egg_Group"), col("Tier_Score")).where(col("Secondary Egg Group").isNotNull())

df_egg = df_egg1.union(df_egg2).groupBy("Egg_Group").avg("Tier_Score")
df_egg.show()

spark.stop()