import os
import sys
import numpy as np
from pyspark import SparkConf
from pyspark.ml.connect import Pipeline
from pyspark.ml.stat import Correlation
from pyspark.shell import sqlContext
from pyspark.sql.functions import col, udf, monotonically_increasing_id, when, log, lit, avg, array, row_number, \
    concat_ws, collect_list, count, mean, unix_timestamp, substring, lower, regexp_replace, expr
from pyspark.sql import SparkSession, Window
from pyspark.ml.feature import VectorAssembler, PCA, StandardScaler, CountVectorizer, StopWordsRemover
from pyspark.ml.clustering import KMeans
from pyspark.ml.classification import DecisionTreeClassifier, NaiveBayes, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, ClusteringEvaluator
from pyspark.ml.feature import StringIndexer
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql.types import IntegerType, DoubleType, StringType
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, split
from pyspark.ml.feature import Tokenizer, HashingTF, IDF, VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import Word2Vec
import seaborn as sns

def iris_clustering():
    spark = SparkSession.builder.appName("Lab2").getOrCreate()
    data_path = "iris.csv"  # Укажите путь к вашему датасету
    iris_df = spark.read.csv(data_path, header=True, inferSchema=True)
    iris_df.printSchema()

    assembler = VectorAssembler(
        inputCols=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
        outputCol="features"
    )
    iris_transformed = assembler.transform(iris_df)

    kmeans = KMeans(k=3, seed=1, featuresCol="features", predictionCol="prediction")
    model = kmeans.fit(iris_transformed)

    clusters = model.transform(iris_transformed)
    clusters.show()

    evaluator = ClusteringEvaluator(featuresCol="features", predictionCol="prediction", metricName="silhouette", distanceMeasure="squaredEuclidean")
    silhouette_score = evaluator.evaluate(clusters)
    print(f"Silhouette with squared Euclidean distance: {silhouette_score}")

    pca = PCA(k=2, inputCol="features", outputCol="pca_features")
    pca_model = pca.fit(clusters)
    pca_result = pca_model.transform(clusters)

    pca_df = pca_result.select("pca_features", "prediction").toPandas()

    pca_df["pca_x"] = pca_df["pca_features"].apply(lambda x: x[0])
    pca_df["pca_y"] = pca_df["pca_features"].apply(lambda x: x[1])

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(pca_df["pca_x"], pca_df["pca_y"], c=pca_df["prediction"], cmap="viridis")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA of Iris Dataset with KMeans Clusters")
    plt.colorbar(scatter, label="Cluster Label")
    plt.show()


def iris_classification():
    spark = SparkSession.builder.appName("Lab2").getOrCreate()

    data_path = "iris.csv"
    iris_df = spark.read.csv(data_path, header=True, inferSchema=True)

    iris_df.printSchema()

    indexer = StringIndexer(inputCol="Species", outputCol="label")
    iris_indexed = indexer.fit(iris_df).transform(iris_df)

    assembler = VectorAssembler(
        inputCols=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
        outputCol="features"
    )

    iris_final = assembler.transform(iris_indexed)
    train_data, test_data = iris_final.randomSplit([0.7, 0.3])

    dt = DecisionTreeClassifier(featuresCol="features", labelCol="label")
    dt_model = dt.fit(train_data)

    predictions = dt_model.transform(test_data)
    predictions.select("prediction", "label", "features").show()

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print(f"Accuracy of Decision Tree model: {accuracy}")


def youtube_dataset_keywords_analysis():
    spark = SparkSession.builder.appName("Lab2").getOrCreate()
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
    data_path = "youtube_channels_1M_clean.csv"
    df = spark.read.csv(data_path, header=True, inferSchema=True, sep=',', multiLine=True)

    df.printSchema()
    df.show()

    df = df.withColumn("subscriber_count", col("subscriber_count").cast("double"))

    df = df.dropna(subset=["subscriber_count", "keywords"])

    df = df.withColumn("subscriber_category",
                       when(col("subscriber_count") < 1000, "Low")
                       .when((col("subscriber_count") >= 1000) & (col("subscriber_count") < 10000), "Medium")
                       .when((col("subscriber_count") >= 10000) & (col("subscriber_count") < 100000), "High")
                       .otherwise("Very High"))

    indexer = StringIndexer(inputCol="subscriber_category", outputCol="label")
    df = indexer.fit(df).transform(df)
    class_counts = df.groupBy("subscriber_category").count()
    class_counts.show()

    df = df.withColumn("keywords", split(col("keywords"), ","))

    cv = CountVectorizer(inputCol="keywords", outputCol="features", vocabSize=1000000, minDF=2.0)
    cv_model = cv.fit(df)
    df_featurized = cv_model.transform(df)

    train_data, test_data = df_featurized.randomSplit([0.7, 0.3], seed=42)

    nb = NaiveBayes(featuresCol="features", labelCol="label", modelType="multinomial")
    nb_model = nb.fit(train_data)
    predictions = nb_model.transform(test_data)
    predictions.select("features", "label", "prediction").show()

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print(f"Accuracy on test data: {accuracy}")

    feature_probs = nb_model.theta.toArray()
    vocab = cv_model.vocabulary
    exp_feature_probs = np.exp(feature_probs)
    labels_metadata = df.select("subscriber_category").distinct().orderBy("subscriber_category").collect()
    labels = [row["subscriber_category"] for row in labels_metadata]

    num_top_words = 10
    for i, class_label in enumerate(labels):
        class_feature_probs = exp_feature_probs[i]
        top_indices = class_feature_probs.argsort()[-num_top_words:][::-1]
        top_words = [(vocab[j], class_feature_probs[j]) for j in top_indices]
        print(f"Top words for class {class_label}:")
        for word, prob in top_words:
            print(f"Keyword: {word}, Probability: {prob}")


def clustring_embeddings_youtube(data_path: str, num_clusters: int, target: str = "subscriber_count"):
    spark = SparkSession.builder.appName("YouTubeKeywordClustering").getOrCreate()
    df = spark.read.csv(data_path, header=True, inferSchema=True)
    df = df.withColumn(target, col(target).cast("double"))
    df = df.dropna(subset=[target, "keywords"])
    df = df.withColumn("keywords", split(col("keywords"), ","))
    word2vec = Word2Vec(vectorSize=100, inputCol="keywords", outputCol="word_vectors")
    w2v_model = word2vec.fit(df)
    keywords_df = df.select(explode(col("keywords")).alias("keyword")).distinct()
    keywords_df = keywords_df.withColumn("keywords", array(col("keyword")))
    keywords_vectors = w2v_model.transform(keywords_df)

    kmeans = KMeans(k=num_clusters, seed=42, featuresCol="word_vectors", predictionCol="cluster")
    kmeans_model = kmeans.fit(keywords_vectors)
    keywords_clusters = kmeans_model.transform(keywords_vectors)

    df_with_keywords = df.select(col("channel_id"), explode(col("keywords")).alias("keyword"), col(target), col("country"))

    df_with_clusters = df_with_keywords.join(keywords_clusters, on="keyword")
    df_keyword_count = df_with_clusters.groupBy("cluster", "keyword").count()

    window_spec = Window.partitionBy("cluster").orderBy(col("count").desc())
    top_keywords = df_keyword_count.withColumn("rank", row_number().over(window_spec)).filter(col("rank") <= 10).distinct()

    top_keywords_list = top_keywords.groupBy("cluster").agg(
        concat_ws(", ", collect_list("keyword")).alias("top_keywords"))

    cluster_avg_subscribers = df_with_clusters.groupBy("cluster").agg(avg(target).alias(f"avg_{target}"))
    cluster_avg_subscribers = cluster_avg_subscribers.join(top_keywords_list, on="cluster")
    cluster_avg_subscribers = cluster_avg_subscribers.orderBy(col(f"avg_{target}").desc())
    cluster_avg_subscribers.show(20, False)

    keywords_clusters = keywords_clusters.join(cluster_avg_subscribers, on="cluster")

    top_keywords_clusters = keywords_clusters.join(top_keywords.select("keyword", "cluster"), on=["keyword", "cluster"])
    pca = PCA(k=2, inputCol="word_vectors", outputCol="pca_features")
    pca_model = pca.fit(top_keywords_clusters)
    keywords_pca = pca_model.transform(top_keywords_clusters)

    pca_df = keywords_pca.select("keyword", "pca_features", "cluster", f"avg_{target}").toPandas()
    pca_df["pca_x"] = pca_df["pca_features"].apply(lambda x: x[0])
    pca_df["pca_y"] = pca_df["pca_features"].apply(lambda x: x[1])

    pca_df["avg_subscribers"] = pd.to_numeric(pca_df[f"avg_{target}"], errors='coerce')
    pca_df.dropna(subset=[f"avg_{target}"], inplace=True)


    plt.figure(figsize=(14, 8))
    scatter = plt.scatter(pca_df["pca_x"], pca_df["pca_y"], c=pca_df["cluster"], cmap="viridis", alpha=0.6,
                          edgecolors='w', linewidth=0.5)

    for i in range(pca_df.shape[0]):
        plt.text(pca_df["pca_x"][i], pca_df["pca_y"][i], pca_df["keyword"][i], fontsize=8, alpha=0.7)

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("PCA of Top 10 Keyword Clusters")
    plt.show()

    spark.stop()
    return df_with_clusters


def analyze_fraud_dataset(data_path: str):

    spark = SparkSession.builder.appName("Lab12").getOrCreate()

    df = spark.read.csv(data_path, header=True, inferSchema=True)

    df = df.dropna()

    feature_cols = [col for col in df.columns if col not in ["Class"]]
    target_col = "Class"

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df_features = assembler.transform(df)

    scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
    df_scaled = scaler.fit(df_features).transform(df_features)

    train_data, test_data = df_scaled.randomSplit([0.7, 0.3], seed=42)

    # ------------------------ Оценка важности признаков ------------------------
    lr = LinearRegression(featuresCol="scaled_features", labelCol=target_col)
    lr_model = lr.fit(train_data)
    feature_importance = sorted(zip(feature_cols, lr_model.coefficients), key=lambda x: abs(x[1]), reverse=True)

    print("Feature Importance:")
    for feature, importance in feature_importance:
        print(f"{feature}: {importance}")

    # ------------------------ Модель кластеризации ------------------------
    kmeans = KMeans(featuresCol="scaled_features", k=3, seed=42)
    kmeans_model = kmeans.fit(df_scaled)
    clusters = kmeans_model.transform(df_scaled)

    pca = PCA(k=2, inputCol="scaled_features", outputCol="pca_features")
    pca_model = pca.fit(clusters)
    pca_result = pca_model.transform(clusters)

    pca_df = pca_result.select("pca_features", "prediction").toPandas()
    pca_df["pca_x"] = pca_df["pca_features"].apply(lambda x: x[0])
    pca_df["pca_y"] = pca_df["pca_features"].apply(lambda x: x[1])

    plt.figure(figsize=(14, 8))
    scatter = plt.scatter(pca_df["pca_x"], pca_df["pca_y"], c=pca_df["prediction"], cmap="viridis", alpha=0.6, edgecolors='w', linewidth=0.5)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("PCA of Clusters")
    plt.colorbar(scatter, label="Cluster")
    plt.show()

    # ------------------------ Модель RandomForest ------------------------
    rf = RandomForestClassifier(featuresCol="scaled_features", labelCol=target_col, numTrees=100)
    rf_model = rf.fit(train_data)
    predictions = rf_model.transform(test_data)

    evaluator = MulticlassClassificationEvaluator(labelCol=target_col, predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print(f"RandomForest Accuracy: {accuracy:.8f}")

    f1_evaluator = MulticlassClassificationEvaluator(labelCol=target_col, predictionCol="prediction", metricName="f1")
    f1_score = f1_evaluator.evaluate(predictions)
    print(f"RandomForest F1-Score: {f1_score:.8f}")
    # Остановка Spark сессии
    spark.stop()

    return feature_importance, accuracy


def analyze_bikes():
    conf = SparkConf().set("spark.driver.memory", "4g").set("spark.executor.memory", "4g")
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")


    data_path_station = "nomenclature_2024.csv"  # Укажите путь к вашему датасету
    stations_df = spark.read.csv(data_path_station, header=True, inferSchema=True)

    data_path_trips = "mibici_2014-2024.csv"  # Укажите путь к вашему датасету
    trips_df = spark.read.csv(data_path_trips, header=True, inferSchema=True)
    trips_df = trips_df.withColumn(
        "Duration",
        (unix_timestamp(substring(col("Duration"), 9, 8), "HH:mm:ss") -
         unix_timestamp(lit("00:00:00"), "HH:mm:ss")) / 60
    )

    # ------------------------ Расчет среднего времени поездки по возрасту и полу ------------------------
    avg_duration_by_age_and_sex = trips_df.groupBy("Age", "Sex").agg(mean("Duration").alias("avg_duration"))
    avg_duration_by_age_and_sex.show()
    avg_duration_pd = avg_duration_by_age_and_sex.toPandas()

    plt.figure(figsize=(14, 8))

    sns.barplot(data=avg_duration_pd, x='Age', y='avg_duration', hue='Sex')

    plt.xlabel('Age')
    plt.ylabel('Number of Users')
    plt.title('Age Distribution by Sex')
    plt.legend(title='Sex')
    plt.grid(True)
    plt.tight_layout()

    plt.show()
    # ------------------------ Кластеризация станций по географическим координатам ------------------------

    feature_cols = ["latitude", "longitude"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    station_features = assembler.transform(stations_df)

    kmeans = KMeans(k=4, seed=42, featuresCol="features")
    kmeans_model = kmeans.fit(station_features)
    clustered_stations = kmeans_model.transform(station_features)

    cluster_pd = clustered_stations.select("latitude", "longitude", "prediction").toPandas()

    plt.figure(figsize=(14, 8))
    scatter = plt.scatter(cluster_pd["latitude"], cluster_pd["longitude"], c=cluster_pd["prediction"], cmap="viridis", alpha=0.6, edgecolors='w', linewidth=0.5)
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")
    plt.title("Station Clusters")
    plt.colorbar(scatter, label="Cluster")
    plt.show()
    # ------------------------ Построение модели для предсказания возраста пользователя ------------------------
    trips_df = trips_df.withColumn("SexIndex", when(col("Sex") == "Male", 1).otherwise(0))

    feature_cols = ["SexIndex", "Duration", "Origin_Id", "Destination_Id"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    feature_df = assembler.transform(trips_df)

    scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
    scaled_df = scaler.fit(feature_df).transform(feature_df)

    train_data, test_data = scaled_df.randomSplit([0.7, 0.3], seed=42)

    lr = LinearRegression(featuresCol="features", labelCol="Age")
    lr_model = lr.fit(train_data)
    predictions = lr_model.transform(test_data)
    evaluator = RegressionEvaluator(labelCol="Age", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print(f"Root Mean Squared Error on test data = {rmse}")


def upwork_analysis():
    spark = SparkSession.builder.appName("Lab2").getOrCreate()
    data_path = "all_upwork_jobs_2024-02-07-2024-03-24.csv"
    data02 = spark.read.csv(data_path, header=True, inferSchema=True)

    data_path = "all_upwork_jobs_2024-03-24-2024-05-21.csv"
    data03 = spark.read.csv(data_path, header=True, inferSchema=True)

    data02.printSchema()
    data = data02.union(data03)
    data = data.na.drop(subset=["title"])

    def process_text(data):
        data = data.withColumn("title_clean", lower(col("title")))
        data = data.withColumn("title_clean", regexp_replace(col("title_clean"), "[^a-zA-Z\\s]", ""))
        data = data.withColumn("words", split(col("title_clean"), "\\s+"))
        hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=10000)
        tf_data = hashingTF.transform(data)
        idf = IDF(inputCol="rawFeatures", outputCol="features")
        idf_model = idf.fit(tf_data)
        tfidf_data = idf_model.transform(tf_data)

        return tfidf_data

    def cluster_jobs(data, num_clusters=5):
        kmeans = KMeans(k=num_clusters, seed=1, featuresCol="features", predictionCol="cluster")
        model = kmeans.fit(data)
        predictions = model.transform(data)
        predictions = predictions.withColumnRenamed("cluster", "prediction")

        return predictions

    data = process_text(data)
    data = cluster_jobs(data)

    cluster_examples = data.groupBy("prediction").agg(collect_list("title").alias("examples"))
    cluster_examples = cluster_examples.withColumn("examples", expr("slice(examples, 1, 5)"))

    cluster_examples.show(truncate=False)

    # --------------------- среднее чисор часрв

    hourly_data = data.filter(col("is_hourly") == "true")
    hourly_data = hourly_data.withColumn("hourly_low", col("hourly_low").cast("float"))
    hourly_data = hourly_data.withColumn("hourly_high", col("hourly_high").cast("float"))
    hourly_data = hourly_data.withColumn("average_hourly_rate", (col("hourly_low") + col("hourly_high")) / 2)
    country_avg_rates = hourly_data.groupBy("country").agg(avg("average_hourly_rate").alias("average_hourly_rate"))
    sorted_country_avg_rates = country_avg_rates.orderBy(col("average_hourly_rate").desc())
    sorted_country_avg_rates.show(100, truncate=False)


def analysis_soccer():
    spark = SparkSession.builder \
        .appName("Lab2") \
        .config("spark.master", "local[*]") \
        .config("spark.jars", "C:\Spark\spark-3.5.1-bin-hadoop3\jars\sqlite-jdbc-3.36.0.3.jar") \
        .getOrCreate()

    tables = ['Country', 'League', 'Match', 'Player', 'Player_Attributes', 'Team', 'Team_Attributes']
    data = {}
    for table in tables:
        data[table] = spark.read.format("jdbc") \
            .option("url", f"jdbc:sqlite:database.sqlite") \
            .option("dbtable", table) \
            .option("driver", "org.sqlite.JDBC") \
            .load()
        print(f'Table {table} schema:')
        data[table].printSchema()

    match_df = data['Match']
    team_attributes_df = data['Team_Attributes']
    player_attributes_df = data['Player_Attributes']
    team_attributes_df = team_attributes_df.dropna(
        subset=["buildUpPlaySpeed", "chanceCreationPassing", "defencePressure"])
    player_attributes_df = player_attributes_df.dropna(
        subset=["overall_rating", "potential", "crossing", "finishing"])
    # ------------------ Кластеризация команд по их характеристикам
    feature_columns = ["buildUpPlaySpeed", "chanceCreationPassing", "defencePressure"]
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    assembled_df = assembler.transform(team_attributes_df)

    scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
    scaler_model = scaler.fit(assembled_df)
    scaled_data = scaler_model.transform(assembled_df)

    kmeans = KMeans(featuresCol="scaled_features", k=3)
    model = kmeans.fit(scaled_data)
    clusters = model.transform(scaled_data)

    pca = PCA(k=2, inputCol="features", outputCol="pca_features")
    pca_model = pca.fit(clusters)
    pca_result = pca_model.transform(clusters)
    pca_pd = pca_result.select("pca_features", "prediction").toPandas()
    pca_pd['pca_x'] = pca_pd['pca_features'].apply(lambda x: x[0])
    pca_pd['pca_y'] = pca_pd['pca_features'].apply(lambda x: x[1])

    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x="pca_x",
        y="pca_y",
        hue="prediction",
        palette="viridis",
        data=pca_pd,
        legend="full"
    )
    plt.title("K-means Clustering with PCA on Soccer Data")
    plt.xlabel("PCA Feature 1")
    plt.ylabel("PCA Feature 2")
    plt.legend(title="Cluster")
    plt.show()

    # ----------------- Оценка результатов команды по ее характеристикам
    match_team_attr_df = match_df.join(
        team_attributes_df,
        match_df.home_team_api_id == team_attributes_df.team_api_id,
        "inner"
    ).select(
        "home_team_goal",
        col("buildUpPlaySpeed").alias("home_buildUpPlaySpeed"),
        col("chanceCreationPassing").alias("home_chanceCreationPassing"),
        col("defencePressure").alias("home_defencePressure")
    )

    match_team_attr_df = match_team_attr_df.dropna()

    assembler = VectorAssembler(
        inputCols=["home_buildUpPlaySpeed", "home_chanceCreationPassing", "home_defencePressure"],
        outputCol="features"
    )
    assembled_df = assembler.transform(match_team_attr_df)

    train_data, test_data = assembled_df.randomSplit([0.7, 0.3], seed=1234)

    lr = LinearRegression(featuresCol="features", labelCol="home_team_goal")
    lr_model = lr.fit(train_data)

    predictions = lr_model.transform(test_data)
    predictions.select("prediction", "home_team_goal").show(10)

    evaluator = RegressionEvaluator(labelCol="home_team_goal", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print(f"Root Mean Squared Error (RMSE) on test data = {rmse}")
    feature_importance = sorted(zip(["home_buildUpPlaySpeed", "home_chanceCreationPassing", "home_defencePressure"], lr_model.coefficients), key=lambda x: abs(x[1]), reverse=True)

    print("Feature Importance:")
    for feature, importance in feature_importance:
        print(f"{feature}: {importance}")
    # ------------------------ Модель для предсказания рейтинга игрока
    feature_columns = [
        "potential", "crossing", "finishing", "short_passing", "dribbling",
        "long_passing", "ball_control", "acceleration", "sprint_speed", "agility"
    ]
    for column in feature_columns:
        mean_value = player_attributes_df.select(mean(col(column))).collect()[0][0]
        player_attributes_df = player_attributes_df.na.fill({column: mean_value})

    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    assembled_df = assembler.transform(player_attributes_df)

    lr = LinearRegression(featuresCol="features", labelCol="overall_rating")
    lr_model = lr.fit(assembled_df)
    feature_importance = sorted(zip(feature_columns, lr_model.coefficients), key=lambda x: abs(x[1]), reverse=True)

    print("Feature Importance:")
    for feature, importance in feature_importance:
        print(f"{feature}: {importance}")


if __name__ == "__main__":
    # iris_clustering()
    # iris_classification()
    # youtube_dataset_keywords_analysis()

    # clustring_embeddings_youtube("youtube_channels_1M_clean.csv", 20)
    # clustring_embeddings_youtube("youtube_channels_1M_clean.csv", 20, target="total_views")
    # analyze_fraud_dataset("creditcard.csv")
    # analyze_bikes()
    # upwork_analysis()
    analysis_soccer()

