# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

from matplotlib.image import imread

import seaborn as sns

from sklearn.datasets.samples_generator import (make_blobs,

                                                make_circles,

                                                make_moons)

from sklearn.cluster import KMeans, SpectralClustering

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import silhouette_samples, silhouette_score

%matplotlib inline
data = pd.read_csv('/kaggle/input/ab-nyc-2019v2/AB_NYC_2019V2.csv')

data.head()
data.shape
plt.figure(figsize=(15,15))

plt.scatter(data.loc[:, "price"], data.loc[:, "number_of_reviews"])

plt.xlabel('Price')

plt.ylabel('Number of Reviews')

plt.title('Price against Number of Reviews')
data = data.fillna(0)

data = data.set_index('neighbourhood_group')
data.tail(20)
df_B = pd.DataFrame(data.loc["Brooklyn", ["price","number_of_reviews"]])

df_M = pd.DataFrame(data.loc["Manhattan", ["price","number_of_reviews"]])

df_Q = pd.DataFrame(data.loc["Queens", ["price","number_of_reviews"]])

df_X = pd.DataFrame(data.loc["Bronx", ["price","number_of_reviews"]])

df_S = pd.DataFrame(data.loc["Staten Island", ["price","number_of_reviews"]])
#https://www.kaggle.com/biphili/hospitality-in-era-of-airbnb

#https://towardsdatascience.com/k-means-clustering-algorithm-applications-evaluation-methods-and-drawbacks-aa03e644b48a

#https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1
X_std_Q = StandardScaler().fit_transform(df_Q)



km_Q = KMeans(n_clusters=5, max_iter=100)

km_Q.fit(X_std_Q)

centroids_Q = km_Q.cluster_centers_



fig, ax = plt.subplots(figsize=(15, 15))

plt.scatter(X_std_Q[km_Q.labels_ == 0, 0], X_std_Q[km_Q.labels_ == 0,1], c ='g', label="Cluster 1")

plt.scatter(X_std_Q[km_Q.labels_ == 1, 0], X_std_Q[km_Q.labels_ == 1,1], c ='k', label="Cluster 2")

plt.scatter(X_std_Q[km_Q.labels_ == 2, 0], X_std_Q[km_Q.labels_ == 2,1], c ='y', label="Cluster 3")

plt.scatter(X_std_Q[km_Q.labels_ == 3, 0], X_std_Q[km_Q.labels_ == 3,1], c ='m', label="Cluster 4")

plt.scatter(X_std_Q[km_Q.labels_ == 4, 0], X_std_Q[km_Q.labels_ == 4,1], c ='c', label="Cluster 5")



plt.scatter(centroids_Q[:, 0], centroids_Q[:, 1], marker ='*', s=300, c = 'r', label='Centroid')



plt.legend()

#plt.xlim([-.5, 30])

#plt.ylim([-.5, 30])

plt.xlabel('Price')

plt.ylabel('Number of Reviews')

plt.title('Quens')

ax.set_aspect('equal')
X_std_M = StandardScaler().fit_transform(df_M)



km_M = KMeans(n_clusters=5, max_iter=100)

km_M.fit(X_std_M)

centroids_M = km_M.cluster_centers_



fig, ax = plt.subplots(figsize=(15, 15))

plt.scatter(X_std_M[km_M.labels_ == 0, 0], X_std_M[km_M.labels_ == 0,1], c ='g', label="Cluster 1")

plt.scatter(X_std_M[km_M.labels_ == 1, 0], X_std_M[km_M.labels_ == 1,1], c ='k', label="Cluster 2")

plt.scatter(X_std_M[km_M.labels_ == 2, 0], X_std_M[km_M.labels_ == 2,1], c ='y', label="Cluster 3")

plt.scatter(X_std_M[km_M.labels_ == 3, 0], X_std_M[km_M.labels_ == 3,1], c ='m', label="Cluster 4")

plt.scatter(X_std_M[km_M.labels_ == 4, 0], X_std_M[km_M.labels_ == 4,1], c ='c', label="Cluster 5")



plt.scatter(centroids_M[:, 0], centroids_M[:, 1], marker ='*', s=300, c = 'r', label='Centroid')



plt.legend()

plt.xlabel('Price')

plt.ylabel('Number of Reviews')

plt.title('Manhattan')

ax.set_aspect('equal')
X_std_B = StandardScaler().fit_transform(df_B)



km_B = KMeans(n_clusters=5, max_iter=100)

km_B.fit(X_std_B)

centroids_B = km_B.cluster_centers_



fig, ax = plt.subplots(figsize=(15, 15))

plt.scatter(X_std_B[km_B.labels_ == 0, 0], X_std_B[km_B.labels_ == 0,1], c ='g', label="Cluster 1")

plt.scatter(X_std_B[km_B.labels_ == 1, 0], X_std_B[km_B.labels_ == 1,1], c ='k', label="Cluster 2")

plt.scatter(X_std_B[km_B.labels_ == 2, 0], X_std_B[km_B.labels_ == 2,1], c ='y', label="Cluster 3")

plt.scatter(X_std_B[km_B.labels_ == 3, 0], X_std_B[km_B.labels_ == 3,1], c ='m', label="Cluster 4")

plt.scatter(X_std_B[km_B.labels_ == 4, 0], X_std_B[km_B.labels_ == 4,1], c ='c', label="Cluster 5")



plt.scatter(centroids_B[:, 0], centroids_B[:, 1], marker ='*', s=300, c = 'r', label='Centroid')



plt.legend()

plt.xlabel('Price')

plt.ylabel('Number of Reviews')

plt.title('Brooklyn')

ax.set_aspect('equal')
X_std_X = StandardScaler().fit_transform(df_X)



km_X = KMeans(n_clusters=5, max_iter=100)

km_X.fit(X_std_X)

centroids_X = km_X.cluster_centers_



fig, ax = plt.subplots(figsize=(15, 15))

plt.scatter(X_std_X[km_X.labels_ == 0, 0], X_std_X[km_X.labels_ == 0,1], c ='g', label="Cluster 1")

plt.scatter(X_std_X[km_X.labels_ == 1, 0], X_std_X[km_X.labels_ == 1,1], c ='k', label="Cluster 2")

plt.scatter(X_std_X[km_X.labels_ == 2, 0], X_std_X[km_X.labels_ == 2,1], c ='y', label="Cluster 3")

plt.scatter(X_std_X[km_X.labels_ == 3, 0], X_std_X[km_X.labels_ == 3,1], c ='m', label="Cluster 4")

plt.scatter(X_std_X[km_X.labels_ == 4, 0], X_std_X[km_X.labels_ == 4,1], c ='c', label="Cluster 5")



plt.scatter(centroids_X[:, 0], centroids_X[:, 1], marker ='*', s=300, c = 'r', label='Centroid')



plt.legend()

plt.xlabel('Price')

plt.ylabel('Number of Reviews')

plt.title('Bronx')

ax.set_aspect('equal')
X_std_S = StandardScaler().fit_transform(df_S)



km_S = KMeans(n_clusters=5, max_iter=100)

km_S.fit(X_std_S)

centroids_S = km_S.cluster_centers_



fig, ax = plt.subplots(figsize=(15, 15))

plt.scatter(X_std_S[km_S.labels_ == 0, 0], X_std_S[km_S.labels_ == 0,1], c ='g', label="Cluster 1")

plt.scatter(X_std_S[km_S.labels_ == 1, 0], X_std_S[km_S.labels_ == 1,1], c ='k', label="Cluster 2")

plt.scatter(X_std_S[km_S.labels_ == 2, 0], X_std_S[km_S.labels_ == 2,1], c ='y', label="Cluster 3")

plt.scatter(X_std_S[km_S.labels_ == 3, 0], X_std_S[km_S.labels_ == 3,1], c ='m', label="Cluster 4")

plt.scatter(X_std_S[km_S.labels_ == 4, 0], X_std_S[km_S.labels_ == 4,1], c ='c', label="Cluster 5")



plt.scatter(centroids_S[:, 0], centroids_S[:, 1], marker ='*', s=300, c = 'r', label='Centroid')



plt.legend()

plt.xlabel('Price')

plt.ylabel('Number of Reviews')

plt.title('Staten Island')

ax.set_aspect('equal')