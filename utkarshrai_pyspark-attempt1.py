!pip install pyspark
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from pyspark.context import SparkContext

from pyspark.sql.context import SQLContext

from pyspark.sql.session import SparkSession

   

sc = SparkContext()

sqlContext = SQLContext(sc)

spark = SparkSession(sc)



# load up other dependencies

import re

import pandas as pd
logfile='/kaggle/input/access-log/access_log'

base_df = spark.read.text(logfile)

base_df.printSchema()
base_df.show(10, truncate=False)

print((base_df.count(), len(base_df.columns)))
from pyspark.sql.functions import regexp_extract



LOG_PATTERN = '^(\S+) (\S+) (\S+) \[([\w:/]+\s[+\-]\d{4})\] "(\S+) (\S+) (\S+)" (\d{3}) (\d+)'



logs_df = base_df.select(regexp_extract('value', LOG_PATTERN, 1).alias('ip_address'),

                         regexp_extract('value', LOG_PATTERN, 2).alias('client_identd'),

                         regexp_extract('value', LOG_PATTERN, 3).alias('user_id'),

                         regexp_extract('value', LOG_PATTERN, 4).alias('timestamp'),

                         regexp_extract('value', LOG_PATTERN, 5).alias('method'),

                         regexp_extract('value', LOG_PATTERN, 6).alias('endpoint'),

                         regexp_extract('value', LOG_PATTERN, 7).alias('protocol'),

                         regexp_extract('value', LOG_PATTERN, 8).cast('integer').alias('status'),

                         regexp_extract('value', LOG_PATTERN, 9).cast('integer').alias('content_size'))

logs_df.show(10, truncate=True)

print((logs_df.count(), len(logs_df.columns)))
status_freq_df = (logs_df

                     .groupBy('status')

                     .count()

                     .sort('status')

                     .cache())

print('Total distinct HTTP Status Codes:', status_freq_df.count())  
status_freq_pd_df = (status_freq_df

                         .toPandas()

                         .sort_values(by=['count'],

                                      ascending=False))

status_freq_pd_df
import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

%matplotlib inline



sns.catplot(x='status', y='count', data=status_freq_pd_df,

            kind='bar', order=status_freq_pd_df['status'])