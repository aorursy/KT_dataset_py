# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

%matplotlib inline

from sklearn import preprocessing as pp

from sklearn.cluster import KMeans

import random 

from sklearn.datasets.samples_generator import make_blobs 

import pylab as pl

%matplotlib inline

card = pd.read_csv("../input/CreditCardUsage.csv")
card.head(5)
card.describe().T
card.isna().sum()
mean_value=card['CREDIT_LIMIT'].mean()

card['CREDIT_LIMIT']=card['CREDIT_LIMIT'].fillna(mean_value)
mean_value=card['MINIMUM_PAYMENTS'].mean()

card['MINIMUM_PAYMENTS']=card['MINIMUM_PAYMENTS'].fillna(mean_value)
card.corr()
card.cov()
a = card.corr()
plt.figure(figsize=(20,10))

sns.heatmap(a,vmin=-1,vmax=1,center=0,annot=True)
df = card.drop('CUST_ID', axis=1)

df.head(3)
from sklearn.preprocessing import StandardScaler

X = df.values[:,1:]

X = np.nan_to_num(X)

Clus_dataSet = StandardScaler().fit_transform(X)

Clus_dataSet
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=6,random_state=0)

kmeans.fit(df)
kmeans.labels_
Sum_of_squared_distances = []

K = range(1,21)

for k in K:

    km = KMeans(n_clusters=k)

    km = km.fit(df)

    Sum_of_squared_distances.append(km.inertia_)

plt.plot(K, Sum_of_squared_distances, 'bx-')

plt.xlabel('k')

plt.ylabel('Sum_of_squared_distances')

plt.title('Elbow Method For Optimal k')

plt.show()
from sklearn.metrics import silhouette_score, silhouette_samples



for n_clusters in range(2,21):

    km = KMeans (n_clusters=n_clusters)

    preds = km.fit_predict(df)

    centers = km.cluster_centers_



    score = silhouette_score(df, preds, metric='euclidean')

    print ("For n_clusters = {}, silhouette score is {}".format(n_clusters, score))
from yellowbrick.cluster import SilhouetteVisualizer



# Instantiate the clustering model and visualizer

km = KMeans (n_clusters=3)

visualizer = SilhouetteVisualizer(km)



visualizer.fit(df) # Fit the training data to the visualizer

visualizer.poof() # Draw/show/poof the data
from yellowbrick.cluster import KElbowVisualizer

# Instantiate the clustering model and visualizer

km = KMeans (n_clusters=3)

visualizer = KElbowVisualizer(

    km, k=(2,21),metric ='silhouette', timings=False

)



visualizer.fit(df) # Fit the training data to the visualizer

visualizer.poof() # Draw/show/poof the data
km_sample = KMeans(n_clusters=4)

km_sample.fit(df)
labels_sample = km_sample.labels_
df['label'] = labels_sample
sns.set_palette('Set2')

sns.scatterplot(df['BALANCE'],df['PURCHASES'],hue=df['label'],palette='Set1')