from pyspark.sql import SparkSession
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lag, lead, when, sum as spark_sum, row_number
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType, FloatType
import pyspark.sql.functions as F

spark = SparkSession.builder \
    .appName("Task3") \
    .getOrCreate()

base_directory = r"C:\Users\elior\OneDrive\Desktop\dataset\ITMO-HSE-MLBD-LW-3"


file_paths = []
for root, dirs, files in os.walk(base_directory):
    for file in files:
        if file == "log.csv":
            file_paths.append(os.path.join(root, file))

print(file_paths)
df = spark.read.load(file_paths, format="csv", inferSchema="true", header="true")

df = df.withColumn("time", df["time"].cast(FloatType()))
df = df.withColumn("cls", df["cls"].cast(IntegerType()))
df = df.orderBy("folder_name", "shot_id", "time")
df.printSchema()
print(df.count())

window_spec = Window.partitionBy("folder_name", "shot_id").orderBy("time")

df = df.withColumn("next_cls", F.lead("cls", 1).over(window_spec))
df = df.withColumn("prev_cls", F.lag("cls", 1).over(window_spec))
df = df.withColumn("next2_cls", F.lead("cls", 2).over(window_spec))
df = df.withColumn("prev2_cls", F.lag("cls", 2).over(window_spec))
df = df.withColumn("next3_cls", F.lead("cls", 3).over(window_spec))
df = df.withColumn("prev3_cls", F.lag("cls", 3).over(window_spec))
df = df.withColumn("next4_cls", F.lead("cls", 4).over(window_spec))
df = df.withColumn("prev4_cls", F.lag("cls", 4).over(window_spec))


df = df.withColumn("cls", F.when(
    (F.col("cls") == 1) & (
        (F.col("next_cls") == 1) |
        (F.col("next2_cls") == 1) |
        (F.col("next3_cls") == 1) |
        (F.col("next4_cls") == 1)), 1
).otherwise(F.col("cls")))


df = df.withColumn("keep", F.when(
    (F.col("cls") == 2) & (F.lag("cls", 1).over(window_spec) == 1) & (F.lead("cls", 1).over(window_spec) != 0), False
).otherwise(True))

df = df.filter(F.col("keep") == True).drop("keep")


df = df.withColumn("group", F.when(
    (F.col("cls") == 1) & ((F.lag("cls", 1).over(window_spec) == 1) | (F.lag("cls", 1).over(window_spec).isNull())), 0
).otherwise(F.monotonically_increasing_id()))

df = df.withColumn("group", F.sum("group").over(window_spec))


df = df.groupBy("folder_name", "shot_id", "group").agg(
    F.first("time").alias("time"),
    F.first("cls").alias("cls")
).orderBy("time")


df = df.withColumn("miss", F.when(
    (F.col("cls") == 0) &
    ((F.lead("cls", 1).over(window_spec) == 2) &
     (F.lead("cls", 2).over(window_spec) == 2) &
     (F.lead("cls", 3).over(window_spec) == 2) ), 1
).otherwise(0))


df = df.withColumn("is_last_row", F.row_number().over(Window.partitionBy("folder_name", "shot_id").orderBy(F.desc("time"))) == 1)
df = df.withColumn("miss", F.when((F.col("cls") == 0) & (F.col("is_last_row") == True), 1).otherwise(F.col("miss")))


df = df.withColumn("miss_group", F.sum("miss").over(Window.orderBy('folder_name', 'shot_id', 'time').rowsBetween(Window.unboundedPreceding, 0)))


hit_counts = df.filter(F.col("cls") == 1).groupBy("miss_group").agg(
    F.count("*").alias("hit_count"),
    F.first("time").alias("start_time"),
    F.first("folder_name").alias("file_name"),
    F.first("shot_id").alias("shot_id"),
    F.last("shot_id").alias("last_shot_id")
)


max_hits = hit_counts.orderBy(F.desc("hit_count")).limit(1).collect()[0]

print(f"Максимальное число попаданий между промахами: {max_hits}")

df.filter(col('folder_name') == '02-07-2024').show(500)