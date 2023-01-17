!pip install pyspark
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))



%env JOBLIB_TEMP_FOLDER=/tmp 

#https://www.kaggle.com/getting-started/45288 - this helps some with 'no space left on device'
import pyspark.sql.functions as sql_func

from pyspark.sql.types import *

from pyspark.ml.recommendation import ALS, ALSModel

from pyspark.context import SparkContext

from pyspark.sql.session import SparkSession



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

    '../input/train_data_full.csv', header=True, schema=data_schema

).cache()
ratings = (final_stat

    .select(

        'user_id',

        'primary_video_id',

        'watching_percentage',

    )

).cache()
%%time

ratings.count()
import gc #This is to free up the memory

gc.collect()

gc.collect()
%%time

als = ALS(rank=100, #rank s the number of latent factors in the model (defaults to 10). Higher value - better accuracy (at this competition), longer training

          maxIter=2, #maxIter is the maximum number of iterations to run (defaults to 10). Higher value - more memory used

          implicitPrefs=True, #implicitPrefs specifies whether to use the explicit feedback ALS variant or one adapted for implicit feedback data (defaults to false)

          regParam=1, #regParam specifies the regularization parameter in ALS (defaults to 1.0)

          alpha=50, #alpha is a parameter applicable to the implicit feedback variant of ALS that governs the baseline confidence in preference observations (defaults to 1.0)

          userCol="user_id", itemCol="primary_video_id", ratingCol="watching_percentage",

          numUserBlocks=32, numItemBlocks=32,

          coldStartStrategy="drop")



model = als.fit(ratings)
%%time

userRecsDf = model.recommendForAllUsers(10).cache()

userRecsDf.count()
userRecs = userRecsDf.toPandas()

userRecs.shape
userRecs[:2]
predicted_dict = userRecs.set_index('user_id').to_dict('index')

predicted_dict = {user_id:[r[0] for r in recs['recommendations']] for user_id, recs in predicted_dict.items()}

len(predicted_dict)
sample_submission = pd.read_csv('../input/sample_submission_full.csv')
sample_submission['als_predicted_primary_video_id'] = sample_submission.user_id.apply(

    lambda user_id: ' '.join([str(v) for v in predicted_dict[user_id]]) if user_id in predicted_dict else None)
sample_submission[:5]
sample_submission['primary_video_id'] = sample_submission.als_predicted_primary_video_id.combine_first(

    sample_submission.primary_video_id)

del sample_submission['als_predicted_primary_video_id']
sample_submission.to_csv('sample_submission_full_als.csv',

                         header=True, index=False)
train_data = pd.read_csv('../input/train_data_full.csv')

train_needed_users =  train_data[train_data.user_id.isin(sample_submission.user_id)]

users_with_history = list(set(train_needed_users.user_id))

cold_users = list(set(sample_submission.user_id) - set(users_with_history))

print('number of users presented in history: ', len(users_with_history), ' % of users with hist data: ',len(users_with_history)/len(sample_submission.user_id))

print('number of cold start users ', len(cold_users), ' % of users without hist data: ', len(cold_users)/len(sample_submission.user_id))
top_10_videos = train_data[train_data['watching_percentage']>=0.5].loc[train_data.session_start_datetime >= '2018-09-20 00:00:00', # Supposing that 10 days closest to testing period is most representative

                               'primary_video_id'].value_counts()[:10].index.tolist()
sample2 = sample_submission.copy()

sample2['forcold'] = ' '.join([str(v) for v in top_10_videos])
sample2.loc[sample2.user_id.isin(cold_users),'primary_video_id'] = sample2['forcold'][1]
del sample2['forcold']
sample2.head()
sample2.to_csv('sparkasl_withcold.csv', #forming another file, with basic processing of cold users 

                         header=True, index=False)