

import findspark
findspark.init()

import pyspark # only run after findspark.init()
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

from pyspark import SparkContext, SparkConf

conf = pyspark.SparkConf().setAppName('covid19 data').setMaster('local')
sc = pyspark.SparkContext.getOrCreate(conf) #(conf=conf)
#spark = SparkSession(sc)
#### import data and gain knowledge about data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
%matplotlib inline

# read the global data
data_path = "../Data/csse_covid_19_time_series/"
confirmed_cases = "../input/covid-data/time_series_covid19_confirmed_global.csv"           #data_path + 'time_series_covid19_confirmed_global.csv'
death_cases ="../input/covid-data/time_series_covid19_deaths_global.csv"                   #data_path + 'time_series_covid19_deaths_global.csv'
recovery_casese ="../input/covid-data/time_series_covid19_recovered_global.csv"            #data_path + 'time_series_covid19_recovered_global.csv'
confirmed_data = pd.read_csv(confirmed_cases)
deaths_data = pd.read_csv(death_cases)
recovered_data = pd.read_csv(recovery_casese)

confirmed_data.head()
confirmed_data.shape
len(confirmed_data["Country/Region"].unique())
confirmed_data.describe()

confirmed_data_spark = spark.read.csv(confirmed_cases,header=True)
deaths_data_spark = spark.read.csv(death_cases,header=True)
recovered_data_spark = spark.read.csv(recovery_casese,header=True)
confirmed_data_spark.select(["Province/State","Country/Region","Lat","Long"]).show(3)
confirmed_data = confirmed_data.drop(["Lat","Long"],axis = 1)
confirmed_data.head(5)
drop_col = ["Lat","Long"]
temp = confirmed_data_spark.select([c for c in confirmed_data_spark.columns if c not in drop_col])
from functools import reduce
from pyspark.sql import DataFrame

confirmed_data_spark = reduce(DataFrame.drop, ['Lat','Long'], confirmed_data_spark)
confirmed_data = confirmed_data.groupby(["Country/Region"]).sum()
confirmed_data.head(5)
sns.heatmap(confirmed_data.isnull())
data_dataype = confirmed_data_spark.dtypes
confirmed_data_spark.groupby().count().show()
confirmed_data_country_wise = confirmed_data_spark.groupby(["Country/Region"]).sum() 
confirmed_data_country_wise.show(5)
from pyspark.sql.types import IntegerType

string_col = ['Province/State','Country/Region']

for col in confirmed_data_spark.columns :
    if col not in string_col :
        confirmed_data_spark = confirmed_data_spark.withColumn(col,confirmed_data_spark[col].cast(IntegerType()))
confirmed_data_country_wise_spark = confirmed_data_spark.groupby(["Country/Region"]).sum() 
confirmed_data_country_wise_spark.count()
confirmed_data.index
confirmed_data.head(5)
confirmed_data = confirmed_data.transpose()
confirmed_data.head()
confirmed_data_rdd = sc.parallelize(confirmed_data)
#confirmed_data_.createOrReplaceTempView("confirmed_data_table")
from pyspark.sql import SQLContext

sqlContext = SQLContext(sc)

confirmed_data_with_country_as_col = sqlContext.createDataFrame(confirmed_data)
len(confirmed_data_with_country_as_col.columns)


plotCountry = ['China', 'US', 'Italy', 'France', 'Spain', 'Australia']
plot_1_datframe = confirmed_data_with_country_as_col.select(plotCountry)
plot_1_datframe.show(5)
#confirmed_data_with_country_as_col.columns
date_of_record = confirmed_data.index

confirmed_data[plotCountry].plot(figsize=(30,15), linewidth=5, colormap='brg', fontsize=25)
#plot_1_datframe.plot(figsize=(30,15), linewidth=5, colormap='brg', fontsize=25)
confirmed_data[plotCountry].plot(figsize=(30,15), linewidth=3, marker='*', colormap='brg', fontsize=25, logy=True)
plt.xlabel('Date', fontsize=25);
plt.ylabel('Confirmed Cases in Logarithmic count', fontsize=25);
plt.title('Confirmed Cases in Logarithmic Time Series', fontsize=25);


ax = confirmed_data.plot(figsize=(30,15), linewidth=2, marker='*', fontsize=25)
ax.legend(ncol=10, loc='upper right')
plt.xlabel('Days', fontsize=20);
plt.ylabel('Number of Reported Confirmed Casese', fontsize=25);
plt.title('Total reported coronavirus casese', fontsize=25);

def max_cases_country_in_give_date(date,data) :
    
    max_index = np.argmax(data[data.index == date].iloc[0].values)
    return data.columns[max_index] 
    
    
max_cases_country_in_give_date("1/24/20",confirmed_data)

confirmed_data.head(5)



deaths_data_spark.columns
drop_col  = ['Lat','Long','Province/State']

deaths_data_spark = deaths_data_spark.select([c for c in deaths_data_spark.columns if c not in drop_col])
len(deaths_data_spark.columns)
deaths_data_spark.count()
temp_dfsp = deaths_data_spark.select('Country/Region','1/22/20')
temp_dfsp.show(5)
deaths_data_spark.createOrReplaceTempView("deaths_data_spark_table")
def max_death_on_given_date(date,deaths_data_spark_table):
    query = "select Country/Region, " + date + " from deaths_data_spark_table "
date = "1/22/20"
query = "select Country/Region, max("+ date +") from ( select Country/Region, " + date + " from deaths_data_spark_table ) "
query
query = " select 1/22/20  from deaths_data_spark_table "
query
#spark.sql(query).show()

deaths_data.head()
deaths_data = deaths_data.drop(["Lat","Long"],axis = 1)
deaths_data.head()
deaths_data = deaths_data.groupby(["Country/Region"]).sum()
deaths_data.head()
deaths_data.info()
deaths_datacopy = deaths_data.copy()
deaths_datacopy.head()


deaths_datacopy = deaths_datacopy.transpose()

ax = deaths_datacopy.plot(figsize=(30,15), linewidth=2, marker='*', fontsize=25)
ax.legend(ncol=10, loc='upper right')
plt.xlabel('Days', fontsize=20);
plt.ylabel('Number of Reported death Casese', fontsize=25);
plt.title('Total reported coronavirus death casese', fontsize=25);
deaths_datacopy.head(5)
deaths_datacopy2 = deaths_datacopy.copy()
def death_more_than30_fun(dataframe) :
    newdf = pd.DataFrame()

    for (val,country) in zip(deaths_datacopy2.iloc[-1],deaths_datacopy2) :
        if(val >= 30) :
            newdf[country] = dataframe[country]
    return newdf  
newdatafram  = death_more_than30_fun(deaths_datacopy2)
newdatafram.head()
newdatafram.shape
ax = newdatafram.plot(figsize=(30,15), linewidth=2, marker='*', fontsize=25)
ax.legend(ncol=10, loc='upper right')
plt.xlabel('Days', fontsize=20);
plt.ylabel('Number of Reported death Casese', fontsize=25);
plt.title('Total reported coronavirus death casese more than 30 in last date', fontsize=25);
