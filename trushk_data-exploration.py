import os

import xgboost as xgb



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()

from haversine import haversine

import datetime as dt





# Load training data as train

train_df = pd.read_csv('../input/train.csv')

test_df  = pd.read_csv('../input/test.csv')



train_df.head(5)
train_df.describe()

test_df.describe()
train_df.get_dtype_counts()

plt.scatter(range(train_df.shape[0]),np.sort(train_df['trip_duration']))

plt.xlabel('Index')

plt.ylabel('Trip Duration')

plt.show()
train_new_df = train_df[train_df.trip_duration < 500000]
plt.scatter(range(train_new_df.shape[0]),np.sort(train_new_df['trip_duration']))

plt.xlabel('Index')

plt.ylabel('Trip Duration')

plt.show()
fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)

long_border = (-74.1, -73.75)

lat_border = (40.6, 40.9)

plt.ylim(lat_border)

plt.xlim(long_border)

plt.xlabel('Longitude')

plt.ylabel('Latitude')

ax[0].scatter(train_new_df.pickup_longitude,train_new_df.pickup_latitude,color='blue',s=1,alpha=0.1)

ax[1].scatter(test_df.pickup_longitude,test_df.pickup_latitude,color='green',s=1,alpha=0.1)

plt.show()
from sklearn.cluster import MiniBatchKMeans



X = np.vstack((train_new_df[['pickup_latitude', 'pickup_longitude']], 

               train_new_df[['dropoff_latitude', 'dropoff_longitude']]))



# Remove abnormal locations

min_lat, min_lng = X.mean(axis=0) - X.std(axis=0)

max_lat, max_lng = X.mean(axis=0) + X.std(axis=0)

X = X[(X[:,0] > min_lat) & (X[:,0] < max_lat) & (X[:,1] > min_lng) & (X[:,1] < max_lng)]



kmeans = MiniBatchKMeans(n_clusters=15, batch_size=32).fit(X)



train_new_df.loc[:,'pickup_cluster']  = kmeans.predict(train_new_df[['pickup_latitude', 'pickup_longitude']])

train_new_df.loc[:,'dropoff_cluster'] = kmeans.predict(train_new_df[['dropoff_latitude', 'dropoff_longitude']])
long_border = (-74.1, -73.75)

lat_border = (40.6, 40.9)

plt.ylim(lat_border)

plt.xlim(long_border)

plt.xlabel('Longitude')

plt.ylabel('Latitude')

scat = plt.scatter(train_new_df.pickup_longitude,train_new_df.pickup_latitude,\

            c=train_new_df.pickup_cluster,label=train_new_df.pickup_cluster,cmap='tab20',s=10, lw=0,)

plt.colorbar(scat)

plt.show()
fig,ax = plt.subplots(ncols=2,sharex=True,sharey=True)

plt.xlabel('Cluster')

plt.ylabel('Trip Duration')

ax[0].hist(train_new_df.pickup_cluster)

ax[1].hist(train_new_df.dropoff_cluster)

ax[0].set_xticks(np.arange(15))

plt.show()

plt.scatter(train_new_df.pickup_cluster,train_new_df.trip_duration)

plt.show()
date_time = pd.to_datetime(train_new_df.pickup_datetime)

train_new_df.loc[:,'weekday'] = date_time.dt.weekday

train_new_df.loc[:,'hour'] = date_time.dt.hour

train_new_df.loc[:,'month'] = date_time.dt.month



plt.scatter(train_new_df.weekday,train_new_df.hour)

plt.xlabel('Weekday')

plt.ylabel('Hour of pickup')

plt.show()



date_time = pd.to_datetime(test_df.pickup_datetime)

test_df.loc[:,'month'] = date_time.dt.month

fig,ax = plt.subplots(ncols=2, sharex=True, sharey=True)



ax[0].hist(train_new_df.month,12,color='blue')

ax[1].hist(test_df.month,12,color='green')

plt.show()
plt.scatter(train_new_df.month,train_new_df.trip_duration)

plt.show()
plt.figure(figsize=(12,9))

sns.heatmap(train_new_df.corr(), vmax=.8, linewidths=0.01,

            square=True,annot=True,cmap='viridis',linecolor="white")

sns.set(font_scale=1)

plt.title('Correlation between features');

plt.show()
