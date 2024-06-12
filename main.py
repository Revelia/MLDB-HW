from dask.dataframe import repartition
from pyspark.sql import SparkSession
from pyspark.sql.functions import expr, col, rand, struct, array, explode, element_at, floor, current_date, date_sub
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, TimestampType
import random, sys, os
from datetime import datetime, timedelta
from pyspark.sql.functions import lit, rand, expr
from pyspark.sql.functions import expr, monotonically_increasing_id, explode, rand, hash, abs
from pyspark.sql.functions import rand, when, col
from itertools import product
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as _sum, collect_list, struct
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, dayofmonth, month, year
from pyspark.sql.types import IntegerType, LongType
from pyspark.sql import functions as F

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date
import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as _sum, collect_list, struct
from pyspark.sql import functions as F


def main():
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
    spark = SparkSession.builder \
        .appName("Lab 1") \
        .master("local[*]") \
        .getOrCreate()

    cities = ["Moscow", "St. Petersburg", "Novosibirsk", "Yekaterinburg", "Nizhny Novgorod",
              "Kazan", "Chelyabinsk", "Omsk", "Samara", "Rostov-on-Don"]

    facility_names = ["PizzaHunt", "KFC", "FixPrice", "HanBuzz"]

    menu_items = ["Pizza", "Tea", "Apple Juice", "Snacks", "Soup"]

    menu_str = ', '.join(['"' + i + '"' for i in menu_items])

    # Генерируем заглушки для заказов
    df = spark.range(100000).withColumn("Order_id", expr("monotonically_increasing_id() + 1"))
    df = df.withColumn("city", expr(f"array({', '.join([repr(c) for c in cities])})[floor(rand() * {len(cities)})]"))
    df = df.withColumn("facility", expr(
        f"array({', '.join([repr(f) for f in facility_names])})[floor(rand() * {len(facility_names)})]"))


    # Генерация DataFrame с уникальными комбинациями города и заведения, добавление координат
    locations = (df.select("city", "facility").distinct()
                 .withColumn("coords", array([struct(
        expr(f"(rand() * 180) - 90").alias("lat"),
        expr(f"(rand() * 360) - 180").alias("lng")
    ) for _ in range(10)])))

    # Объединение координат с основным DataFrame
    df = df.join(locations, ["city", "facility"])

    # На этом этапе в coords списки json координат
    df.show()

    df = df.withColumn("index", (abs(hash("Order_id")) % 10) + 1)  # Создаем индекс от 1 до 10 на основе Order_id
    df = df.withColumn("selected_coord", element_at(col("coords"), col("index")))
    df = df.withColumn("lat", col("selected_coord.lat"))
    df = df.withColumn("lng", col("selected_coord.lng"))
    df = df.drop("coords", "selected_coord", "index")
    df = df.withColumn("country", lit("Russia"))

    df = df.withColumn("datetime", date_sub(current_date(), expr("cast(floor(rand() * 10) as int)")))

    df = df.withColumn("Items", expr(f"transform(sequence(1, floor(rand() * 6 + 5)), " + \
                                     f"i -> struct(array({menu_str})[floor(rand() * {len(menu_items)})] as menu_item, " + \
                                     "format_number(rand() * 18.5 + 1.5, '0.00') as price))" ))

    # Разделение массива элементов на отдельные строки
    df = df.withColumn("Item", explode("Items")).select("Order_id", "city", "facility", "country", "lat", "lng", "datetime", "Item.*")
    df = df.withColumn("rand", rand())
    df = df.withColumn("lat", when(df["rand"] < 0.1, 999).otherwise(col("lat")))
    df = df.drop("rand")
    df.repartition(10).write.format("com.databricks.spark.csv").option("header", "true").save("output", mode="overwrite")
    df.show(5)
    spark.stop()


def filter_bad_data():
    spark = SparkSession.builder \
        .appName("Data Processing") \
        .config("spark.jars.packages", "org.apache.spark:spark-avro_2.12:3.5.1") \
        .master("local[*]") \
        .getOrCreate()

    # Чтение данных из CSV
    df = spark.read.csv("output", header=True, inferSchema=True)
    df = df.withColumn("Order_id", col("Order_id").cast("long")) \
           .withColumn("lat", col("lat").cast("float")) \
           .withColumn("lng", col("lng").cast("float")) \
           .withColumn("price", col("price").cast("float")) \
           .withColumn("datetime", col("datetime").cast("timestamp"))


    corrupted = df.filter("lat = 999 or lng = 999")
    clean_data = df.filter("lat <> 999 and lng <> 999")

    # Запись испорченных записей в deadletter файл
    corrupted.write.format("csv").option("header", True).mode("overwrite").save("deadletter.csv")

    clean_data = clean_data.withColumn("date", to_date(col("datetime")))

    # Запись корректных данных в Avro и Parquet, с партиционированием по дате и городу
    clean_data.write.format("avro").partitionBy("date", "city").mode("overwrite").save("data.avro")
    clean_data.write.format("parquet").partitionBy("date", "city").mode("overwrite").save("data.parquet")

    spark.stop()

def save_to_base():
    spark = SparkSession.builder \
        .appName("Avro to Postgres") \
        .config("spark.master", "local[*]") \
        .config("spark.jars.packages", "org.apache.spark:spark-avro_2.12:3.5.1") \
        .config("spark.jars", "C:\\path\\to\\postgresql-42.7.3.jar") \
        .getOrCreate()

    df = spark.read.format("avro").load("data.avro")
    url = "jdbc:postgresql://localhost:5432/base"
    properties = {
        "user": "postgres",
        "driver": "org.postgresql.Driver"
    }

    # Запись в таблицу Place
    df.select("Order_id", "lat", "lng").write \
        .jdbc(url, "Place", "overwrite", properties)

    # Запись в таблицу Prices
    df.select("Order_id", "datetime", "menu_item", "price").write \
        .jdbc(url, "Prices", "overwrite", properties)

    # Запись в таблицу City
    df.select("lng", "lat", "city").distinct().write \
        .jdbc(url, "City", "overwrite", properties)

    # Запись в таблицу Country
    df.select("city", "country").distinct().write \
        .jdbc(url, "Country", "overwrite", properties)


def read_from_avro_and_write_to_postgres():
    spark = SparkSession.builder \
        .appName("Avro to Postgres") \
        .config("spark.master", "local[*]") \
        .config("spark.jars.packages", "org.apache.spark:spark-avro_2.12:3.5.1") \
        .config("spark.jars", "C:\\path\\to\\postgresql-42.7.3.jar") \
        .getOrCreate()

    df = spark.read.format("avro").load("data.avro")

    # Создание таблицы измерений dim_location
    dim_location = df.select("city", "facility", "country", "lat", "lng").distinct()
    dim_location = dim_location.withColumn("location_id", F.monotonically_increasing_id())

    # Создание таблицы измерений dim_menu_item
    dim_menu_item = df.select("menu_item", "price").distinct()
    dim_menu_item = dim_menu_item.withColumn("menu_item_id", F.monotonically_increasing_id())

    # Создание таблицы измерений dim_time
    dim_time = df.select("datetime").distinct()
    dim_time = dim_time.withColumn("time_id", F.monotonically_increasing_id()) \
                       .withColumn("day", dayofmonth("datetime")) \
                       .withColumn("month", month("datetime")) \
                       .withColumn("year", year("datetime"))

    fact_orders = df.join(dim_location, ["city", "facility", "country", "lat", "lng"], "left") \
                    .join(dim_menu_item, ["menu_item", "price"], "left") \
                    .join(dim_time, ["datetime"], "left") \
                    .select(
                        col("Order_id").alias("order_id"),
                        col("menu_item_id"),
                        col("location_id"),
                        col("time_id")
                    )

    # Параметры подключения к Postgres
    postgres_url = "jdbc:postgresql://localhost:5432/star_base"
    properties = {
        "user": "postgres",
        "driver": "org.postgresql.Driver"
    }

    # Запись таблиц в Postgres
    dim_location.write.jdbc(postgres_url, "dim_location", mode="overwrite", properties=properties)
    dim_menu_item.write.jdbc(postgres_url, "dim_menu_item", mode="overwrite", properties=properties)
    dim_time.write.jdbc(postgres_url, "dim_time", mode="overwrite", properties=properties)
    fact_orders.write.jdbc(postgres_url, "fact_orders", mode="overwrite", properties=properties)

    # Остановка SparkSession
    spark.stop()




def mongo_task():
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

    spark = SparkSession.builder \
        .appName("Parquet to MongoDB") \
        .config("spark.master", "local[*]") \
        .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1,org.apache.spark:spark-avro_2.12:3.5.1") \
        .config("spark.mongodb.input.uri", "mongodb://localhost:27017/admin.base1") \
        .config("spark.mongodb.output.uri", "mongodb://localhost:27017/admin.base1") \
        .getOrCreate()
    schema = StructType([
        StructField("Order_id", LongType(), True),
        StructField("city", StringType(), True),
        StructField("facility", StringType(), True),
        StructField("country", StringType(), True),
        StructField("lat", FloatType(), True),
        StructField("lng", FloatType(), True),
        StructField("menu_item", StringType(), True),
        StructField("price", FloatType(), True),
        StructField("datetime", TimestampType(), True)
    ])
    df = spark.read.schema(schema).parquet("data.parquet")
    orders = df.groupBy("Order_id").agg(
        _sum("price").alias("total_amount"),  # Сумма заказа
        collect_list(
            struct(
                col("menu_item").alias("menu_item"),
                col("price").alias("price"),
                col("datetime").alias("datetime")
            )
        ).alias("order_items")  # Массив позиций заказа
    )

    orders.write.format("mongo").mode("overwrite").save()


    spark.stop()


if __name__ == "__main__":

    # main()
    # filter_bad_data()
    # save_to_base()
    # read_from_avro_and_write_to_postgres()
    mongo_task()
