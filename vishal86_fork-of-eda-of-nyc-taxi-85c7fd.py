#installing Tensorflow for future use 

import tensorflow as tf

tf.reset_default_graph()
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

% matplotlib inline





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
NYC_Taxi_train=pd.read_csv('../input/train.csv')

NYC_Taxi_test=pd.read_csv('../input/test.csv')
NYC_Taxi_train.head()
NYC_Taxi_train.columns
#Data fields



# id - a unique identifier for each trip

# vendor_id - a code indicating the provider associated with the trip record

# pickup_datetime - date and time when the meter was engaged

# dropoff_datetime - date and time when the meter was disengaged

# passenger_count - the number of passengers in the vehicle (driver entered value)

# pickup_longitude - the longitude where the meter was engaged

# pickup_latitude - the latitude where the meter was engaged

# dropoff_longitude - the longitude where the meter was disengaged

# dropoff_latitude - the latitude where the meter was disengaged

# store_and_fwd_flag - This flag indicates whether the trip record was held in vehicle memory before sending to the vendor because the vehicle did not have a connection to the server - Y=store and forward; N=not a store and forward trip

# trip_duration - duration of the trip in seconds
#univariate analysis (analysis of all features individually )
# Obervation with id

# checking duplicate with 'id' feature 

NYC_Taxi_train['id'].duplicated().value_counts()

NYC_Taxi_test['id'].duplicated().value_counts()

# unique id count in train:1458644

# unique id count in test: 625134



# Obervation with vendor id

NYC_Taxi_train['vendor_id'].duplicated().value_counts()

NYC_Taxi_test['vendor_id'].duplicated().value_counts()

NYC_Taxi_train.groupby('vendor_id')['vendor_id'].sum()

# two vendor's provide taxi as per data set in future we would like to explore are they providing taxi's in some perticular let-log(area of NYC)
# popularity of vendor 

NYC_Taxi_train.groupby('vendor_id')['vendor_id'].sum().plot(kind='bar',figsize=(8,6))
# Passenger_count trend 

NYC_Taxi_train['passenger_count'].value_counts().sort_values()

NYC_Taxi_train['passenger_count'].value_counts().sort_values().plot(kind='barh',figsize=(8,6))

# Observations:

# 1) 60 taxi running with out passenger :) 

# 2) mostly passenger travel alone or with one more passenger ,after that thrid largest count is of 5 passenger's in group
# analysis of trip duration 

NYC_Taxi_train['trip_duration'].isnull().value_counts()

NYC_Taxi_train['trip_duration'].max()

NYC_Taxi_train['trip_duration'].min()

# funny minimum trip_duration is 1 sec 
NYC_Taxi_train_alt=NYC_Taxi_train[NYC_Taxi_train['trip_duration']<120]

NYC_Taxi_train_alt['trip_duration'].count()

# total trips finished with in 2 mins = 27817
# create new columns trip duration in mins AND trip duration in hours

NYC_Taxi_train['trip_duration_in_min']=(NYC_Taxi_train['trip_duration']/60).round(1)

NYC_Taxi_train['trip_duration_in_hour']=(NYC_Taxi_train['trip_duration_in_min']/60).round(2)
NYC_Taxi_train['trip_duration_in_min'].mean()
NYC_Taxi_train['trip_duration_in_min'].min()
NYC_Taxi_train['trip_duration_in_min'].max()
# in train dataset some trip duration are very high (I consider them outliers and remove them before replotting it)

q = NYC_Taxi_train.trip_duration.quantile(0.99)

NYC_Taxi_train = NYC_Taxi_train[NYC_Taxi_train.trip_duration < q]

plt.figure(figsize=(8,6))

plt.scatter(range(NYC_Taxi_train.shape[0]), np.sort(NYC_Taxi_train.trip_duration.values))

plt.xlabel('index', fontsize=12)

plt.ylabel('trip duration', fontsize=12)

plt.show()