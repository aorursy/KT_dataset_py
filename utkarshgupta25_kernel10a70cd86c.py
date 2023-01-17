# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/Mall_Customers.csv')



df.head()
df.isna().sum()
df.describe()
from sklearn.cluster import KMeans



fig = plt.figure(figsize = (10,8))



WCSS = []



x = df.drop(columns = ['CustomerID', 'Genre', 'Age']).values



for i in range(1, 11):

    clf = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 17)

    

    clf.fit(x)

    WCSS.append(clf.inertia_)

    

plt.plot(range(1,11), WCSS)

plt.title('The Elbow Method')

plt.ylabel('WCSS')

plt.xlabel('Clusters')

plt.show()
clf = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10,  random_state=0)

y_kmeans = clf.fit_predict(x)



fig = plt.figure(figsize=(10, 8))

plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], color='red', s=60, label='Cluster 1', edgecolors='black')

plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], color='green', s=60, label='Cluster 2', edgecolors='black')

plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], color='blue',s=60, label='Cluster 3', edgecolors='black')

plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], color='yellow', s=60, label='Cluster 4', edgecolors='black')

plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], color='cyan', s=60, label='Cluster 5', edgecolors='black')

# cluster centres

plt.scatter(clf.cluster_centers_[:, 0], clf.cluster_centers_[:, 1], color='magenta', s=100, label='Centroid',edgecolors='black')

plt.legend()

plt.title('Clusters using KMeans')

plt.ylabel('Annual Income (k$)')

plt.xlabel('Spending Score (1-100)')

plt.show() 

# Using Dendrogram to find the optimal number of clusters

import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(x, method='ward')) # The ward method tries to minimise the variance in each cluster



plt.title('Dendogram')

plt.xlabel('Customers')

plt.ylabel('Euclidean Distance')

plt.show()



# Fitting hierarchical clustering model

from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')

y_hc = hc.fit_predict(x)

y_hc





plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], color='red', s=60, label='Cluster 1', edgecolors='black')

plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], color='green', s=60, label='Cluster 2', edgecolors='black')

plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], color='blue',s=60, label='Cluster 3', edgecolors='black')

plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], color='yellow', s=60, label='Cluster 4', edgecolors='black')

plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], color='cyan', s=60, label='Cluster 5', edgecolors='black')

# cluster centres

# plt.scatter(hc.cluster_centers_[:, 0], hc.cluster_centers_[:, 1], color='magenta', s=100, label='Centroid',edgecolors='black')

plt.legend()

plt.title('Hierarchical Clustering')

plt.ylabel('Annual Income (k$)')

plt.xlabel('Spending Score (1-100)')

plt.show()