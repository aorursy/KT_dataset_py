# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

paths = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        paths.append(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install pyspark
import os

import pandas as pd

import numpy as np



from pyspark import SparkConf, SparkContext

from pyspark.sql import SparkSession, SQLContext



from pyspark.sql.types import *

import pyspark.sql.functions as F

from pyspark.sql.functions import *
spark = SparkSession.builder.master("local[2]").appName("BikeSharing").getOrCreate()
spark
sc = spark.sparkContext

sc
sqlContext = SQLContext(spark.sparkContext)

sqlContext
paths = list(set(paths)-set(['/kaggle/input/bixi-montreal-bikeshare-data/BixiMontrealRentals2018/Stations_2018.csv',

                '/kaggle/input/bixi-montreal-bikeshare-data/BixiMontrealRentals2019/Stations_2019.csv',

                '/kaggle/input/bixi-montreal-bikeshare-data/BixiMontrealRentals2017/2017/Stations_2017.csv',

                '/kaggle/input/bixi-montreal-bikeshare-data/BixiMontrealRentals2016/BixiMontrealRentals2016/Stations_2016.csv']))
stations_df_16 = spark.read.format("csv").options(header="true").option("delimiter",",").option("encoding", "UTF-8").option("quote","\"").load('/kaggle/input/bixi-montreal-bikeshare-data/BixiMontrealRentals2016/BixiMontrealRentals2016/Stations_2016.csv').cache()

stations_df_17 = spark.read.format("csv").options(header="true").option("delimiter",",").option("encoding", "UTF-8").option("quote","\"").load('/kaggle/input/bixi-montreal-bikeshare-data/BixiMontrealRentals2017/2017/Stations_2017.csv').cache()

stations_df_18 = spark.read.format("csv").options(header="true").option("delimiter",",").option("encoding", "UTF-8").option("quote","\"").load('/kaggle/input/bixi-montreal-bikeshare-data/BixiMontrealRentals2018/Stations_2018.csv').cache()

stations_df_19 = spark.read.format("csv").options(header="true").option("delimiter",",").option("encoding", "UTF-8").option("quote","\"").load('/kaggle/input/bixi-montreal-bikeshare-data/BixiMontrealRentals2019/Stations_2019.csv').cache()

stations_df_16 = stations_df_16.withColumn("year_station",lit('2016'))

stations_df_17 = stations_df_17.withColumn("year_station",lit('2017'))

stations_df_18 = stations_df_18.withColumn("year_station",lit('2018'))

stations_df_19 = stations_df_19.withColumn("year_station",lit('2019'))

cols = stations_df_16.columns



stations_df = stations_df_16.select(cols).union(stations_df_17.select(cols)).union(stations_df_18.select(cols)).union(stations_df_19.select(cols)).cache()
bike_df_full = spark.read.format("csv").options(header="true").option("delimiter",",").option("encoding", "UTF-8").option("quote","\"").load(paths).cache()
bike_df_full.count()
bike_df_full.columns
bike_df_full.dtypes
bike_df_full.take(5)
bike_df_full.count()
bike_df_full = (bike_df_full.withColumn('year_start_date',year(bike_df_full.start_date))

          .withColumn('month_start_date',month(bike_df_full.start_date))

          .withColumn('hour_start_date',hour(bike_df_full.start_date))

          .withColumn('year_end_date',year(bike_df_full.end_date))

          .withColumn('month_end_date',month(bike_df_full.end_date))

          .withColumn('hour_end_date',hour(bike_df_full.end_date))

          )
for c in bike_df_full.columns:

    print("number of nulls in ", c ,bike_df_full.filter(col(c).isNull()).count())

bike_df_full.take(5)
hourly_start_count = bike_df_full.groupBy(bike_df_full.hour_start_date).count().orderBy(col('hour_start_date'))
hourly_start_count.toPandas().plot.bar(x='hour_start_date',figsize=(14, 6))
hourly_end_count = bike_df_full.groupBy(bike_df_full.hour_end_date).count().orderBy(col('hour_end_date'))
hourly_end_count.toPandas().plot.bar(x='hour_end_date',figsize=(14, 6))
bike_df_full.groupBy(col('is_member')).count().show()
hourly_ismember_end_count = bike_df_full.groupBy(bike_df_full.hour_end_date,bike_df_full.is_member).count().orderBy(col('hour_end_date'))
df_pivot = hourly_ismember_end_count.toPandas().pivot(index='hour_end_date', columns='is_member', values='count')
df_pivot.plot.bar(stacked=True, figsize=(10,7))
bike_df_full.groupBy(col('is_member')).agg({'duration_sec':'sum'}).toPandas().plot.bar(x='is_member')
duration_hour_member = bike_df_full.groupBy(col('is_member'),col('hour_start_date')).agg({'duration_sec':'sum'}).withColumnRenamed('sum(duration_sec)','tot_duration').toPandas().pivot(index='hour_start_date', columns='is_member', values='tot_duration')
duration_hour_member.plot.bar(stacked=True, figsize=(10,7))
bike_df_full.groupBy(col('start_station_code'),col('end_station_code')).agg({'start_station_code':'count','duration_sec':'sum'}).withColumnRenamed('count(start_station_code)','num_of_rides').withColumnRenamed('sum(duration_sec)','tot_duration').filter(col('start_station_code')!= col('end_station_code')).orderBy(desc('tot_duration'), desc('num_of_rides')).show(10)
bike_df_full.groupBy(col('start_station_code'),col('end_station_code')).agg({'start_station_code':'count','duration_sec':'sum'}).withColumnRenamed('count(start_station_code)','num_of_rides').withColumnRenamed('sum(duration_sec)','tot_duration').filter(col('start_station_code')== col('end_station_code')).orderBy(desc('tot_duration'), desc('num_of_rides')).show(10)
stations_df.columns
bike_df_full.columns
bike_df_full.filter(col('year_start_date')!=col('year_end_date')).count()
join_type = "left_outer"



df_final = bike_df_full.join(broadcast(stations_df) , (stations_df.code==bike_df_full.start_station_code) & (stations_df.year_station==bike_df_full.year_end_date),join_type)
df_final =  df_final.cache()
df_final.columns
df_final.count()
df_final = df_final.withColumnRenamed("name","starting_station_name").withColumnRenamed("latitude","starting_station_lat").withColumnRenamed("longitude","starting_station_long").drop('year_station').drop('code')
df_final.columns
df_final = df_final.join(broadcast(stations_df) , (stations_df.code==df_final.end_station_code) & (stations_df.year_station==df_final.year_end_date),join_type)
df_final = df_final.withColumnRenamed("name","ending_station_name").withColumnRenamed("latitude","ending_station_lat").withColumnRenamed("longitude","ending_station_long").drop('year_station').drop('code')
df_final.columns



#df_final.filter(col('start_station_code')!=col('code')).count()
df_map = df_final.groupBy('starting_station_name','starting_station_lat','starting_station_long').agg({'starting_station_name':'count','duration_sec':'sum'}).withColumnRenamed('count(starting_station_name)','num_of_rides').withColumnRenamed('sum(duration_sec)','tot_duration').orderBy(desc('num_of_rides')).limit(10000).toPandas()
df_map["starting_station_lat"] = pd.to_numeric(df_map['starting_station_lat'])

df_map["starting_station_long"] = pd.to_numeric(df_map['starting_station_long'])
import matplotlib.pyplot as plt

df_map.plot(kind="scatter", x="starting_station_long", y="starting_station_lat",

    s=df_map['num_of_rides']/100, label="number of rides",

    c="tot_duration", cmap=plt.get_cmap("jet"),

    colorbar=True, alpha=0.1, figsize=(10,7),

)

plt.legend()

plt.show()