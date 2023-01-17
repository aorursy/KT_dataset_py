import os

import pandas as pd

import numpy as np

from matplotlib.pyplot import *

import matplotlib.pyplot as plt

from matplotlib import animation

from matplotlib import cm

from sklearn.cluster import KMeans

from sklearn.neighbors import KNeighborsClassifier

from dateutil import parser

import io

import base64

from IPython.display import HTML

from imblearn.under_sampling import RandomUnderSampler

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
df = pd.read_csv('../input/train.csv')
df.head()
xlim = [-74.03, -73.77]

ylim = [40.63, 40.85]

df = df[(df.pickup_longitude> xlim[0]) & (df.pickup_longitude < xlim[1])]

df = df[(df.dropoff_longitude> xlim[0]) & (df.dropoff_longitude < xlim[1])]

df = df[(df.pickup_latitude> ylim[0]) & (df.pickup_latitude < ylim[1])]

df = df[(df.dropoff_latitude> ylim[0]) & (df.dropoff_latitude < ylim[1])]
longitude = list(df.pickup_longitude) + list(df.dropoff_longitude)

latitude = list(df.pickup_latitude) + list(df.dropoff_latitude)

plt.figure(figsize = (10,10))

plt.plot(longitude,latitude,'.', alpha = 0.4, markersize = 0.05)

plt.show()
loc_df = pd.DataFrame()

loc_df['longitude'] = longitude

loc_df['latitude'] = latitude
kmeans = KMeans(n_clusters=15, random_state=2, n_init = 10).fit(loc_df)

loc_df['label'] = kmeans.labels_



loc_df = loc_df.sample(200000)

plt.figure(figsize = (10,10))

for label in loc_df.label.unique():

    plt.plot(loc_df.longitude[loc_df.label == label],loc_df.latitude[loc_df.label == label],'.', alpha = 0.3, markersize = 0.3)



plt.title('Clusters of New York')

plt.show()
fig,ax = plt.subplots(figsize = (10,10))

for label in loc_df.label.unique():

    ax.plot(loc_df.longitude[loc_df.label == label],loc_df.latitude[loc_df.label == label],'.', alpha = 0.4, markersize = 0.1, color = 'gray')

    ax.plot(kmeans.cluster_centers_[label,0],kmeans.cluster_centers_[label,1],'o', color = 'r')

    ax.annotate(label, (kmeans.cluster_centers_[label,0],kmeans.cluster_centers_[label,1]), color = 'b', fontsize = 20)

ax.set_title('Cluster Centers')

plt.show()