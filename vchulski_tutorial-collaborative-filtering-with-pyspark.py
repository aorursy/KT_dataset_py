import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import gc



%env JOBLIB_TEMP_FOLDER=/tmp 

#https://www.kaggle.com/getting-started/45288 - this helps some with 'no space left on device'



#print(os.listdir("../input/megogochallenge/"))
full_df = pd.read_csv('../input/megogochallenge/train_data_full.csv')

full_df.head()
full_df['watching_percentage'].hist()
!pip install pyspark
import pyspark.sql.functions as sql_func

from pyspark.sql.types import *

from pyspark.ml.recommendation import ALS, ALSModel

from pyspark.context import SparkContext

from pyspark.sql.session import SparkSession

from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics

from pyspark.ml.evaluation import RegressionEvaluator



sc = SparkContext('local') #https://stackoverflow.com/questions/30763951/spark-context-sc-not-defined

spark = SparkSession(sc)
data_schema = StructType([

    StructField('session_start_datetime',TimestampType(), False),

    StructField('user_id',IntegerType(), False),

    StructField('user_ip',IntegerType(), False),

    StructField('primary_video_id',IntegerType(), False),

    StructField('video_id',IntegerType(), False),

    StructField('vod_type',StringType(), False),

    StructField('session_duration',IntegerType(), False),

    StructField('device_type',StringType(), False),

    StructField('device_os',StringType(), False),

    StructField('player_position_min',LongType(), False),

    StructField('player_position_max',LongType(), False),

    StructField('time_cumsum_max',LongType(), False),

    StructField('video_duration',IntegerType(), False),

    StructField('watching_percentage',FloatType(), False)

])

final_stat = spark.read.csv(

    '../input/megogochallenge/train_data_full.csv', header=True, schema=data_schema

).cache()
ratings = (final_stat

    .select(

        'user_id',

        'primary_video_id',

        'watching_percentage',

    )

).cache()
(training, test) = ratings.randomSplit([0.8, 0.2])
# Build the recommendation model using ALS on the training data

# Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics

als = ALS(maxIter=2, regParam=0.01, 

          userCol="user_id", itemCol="primary_video_id", ratingCol="watching_percentage",

          coldStartStrategy="drop",

          implicitPrefs=True)

model = als.fit(training)



# Evaluate the model by computing the RMSE on the test data

predictions = model.transform(test)

evaluator = RegressionEvaluator(metricName="rmse", labelCol="watching_percentage",

                                predictionCol="prediction")



rmse = evaluator.evaluate(predictions)

print("Root-mean-square error = " + str(rmse))
# Build the recommendation model using ALS on the training data

# Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics

als = ALS(maxIter=2, regParam=0.01, 

          userCol="user_id", itemCol="primary_video_id", ratingCol="watching_percentage",

          coldStartStrategy="drop",

          implicitPrefs=False) #changed param!

model = als.fit(training)



# Evaluate the model by computing the RMSE on the test data

predictions = model.transform(test)

evaluator = RegressionEvaluator(metricName="rmse", labelCol="watching_percentage",

                                predictionCol="prediction")



rmse = evaluator.evaluate(predictions)

print("Root-mean-square error = " + str(rmse))
# Build the recommendation model using ALS on the training data

# Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics

als = ALS(rank=20, #10 was by default

          maxIter=2, regParam=0.01,

          userCol="user_id", itemCol="primary_video_id", ratingCol="watching_percentage",

          coldStartStrategy="drop",

          implicitPrefs=False)

model = als.fit(training)



# Evaluate the model by computing the RMSE on the test data

predictions = model.transform(test)

evaluator = RegressionEvaluator(metricName="rmse", labelCol="watching_percentage",

                                predictionCol="prediction")



rmse = evaluator.evaluate(predictions)

print("Root-mean-square error = " + str(rmse))

%%time

# Generate top 10 movie recommendations for each user

userRecs = model.recommendForAllUsers(10)

userRecs.count()

# Generate top 10 user recommendations for each movie

movieRecs = model.recommendForAllItems(10)

movieRecs.count()
userRecs_df = userRecs.toPandas()

print(userRecs_df.shape)



movieRecs_df = movieRecs.toPandas()

print(movieRecs_df.shape)
userRecs_df.head()
movieRecs_df.head()