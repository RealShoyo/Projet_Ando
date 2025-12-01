from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace


spark = SparkSession.builder.appName("NettoyageGuillemets").getOrCreate()


file_path = "raw/dataset.csv"
df = spark.read.csv(file_path, header=True, inferSchema=True)


print("Schéma initial :")
df.printSchema()


string_cols = [f.name for f in df.schema.fields if f.dataType.typeName() == 'string']


for column in string_cols:
    df = df.withColumn(
        column,
        regexp_replace(col(column), '"""', '') # Remplace '"""' par une chaîne vide ''
    )

print("\nDataFrame après nettoyage des '\"\"\"' :")
df.show(5, truncate=False)


print("\nSchéma final :")
df.printSchema()

df_pandas = df.toPandas()

df_pandas.to_csv("raw/dataset_v2.csv", index=False)

spark.stop()