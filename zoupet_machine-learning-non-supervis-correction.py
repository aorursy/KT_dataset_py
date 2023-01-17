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
from sklearn.preprocessing import StandardScaler



import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline 



import os

import warnings



warnings.filterwarnings('ignore')
df = pd.read_csv('/kaggle/input/ml-training-vlib/Mall_Customers.csv')

df.head()
df.info(), df.describe(), df.shape
df.rename(index=str, columns={'Annual Income (k$)': 'Income',

                              'Spending Score (1-100)': 'Score'}, inplace=True)

df.head()
# Let's see our data in a detailed way with pairplot

X = df.drop(['CustomerID', 'Gender'], axis=1)

sns.pairplot(df.drop('CustomerID', axis=1), hue='Gender', aspect=1.5)

plt.show()
from sklearn.cluster import KMeans



clusters = []



for i in range(1, 11):

    km = KMeans(n_clusters=i).fit(X)

    clusters.append(km.inertia_)

    

fig, ax = plt.subplots(figsize=(12, 8))

sns.lineplot(x=list(range(1, 11)), y=clusters, ax=ax)

ax.set_title('Searching for Elbow')

ax.set_xlabel('Clusters')

ax.set_ylabel('Inertia')



# Annotate arrow

ax.annotate('Possible Elbow Point', xy=(3, 140000), xytext=(3, 50000), xycoords='data',          

             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2))



ax.annotate('Possible Elbow Point', xy=(5, 80000), xytext=(5, 150000), xycoords='data',          

             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2))



plt.show()
# 3 cluster

km3 = KMeans(n_clusters=3).fit(X)



X['Labels'] = km3.labels_

plt.figure(figsize=(12, 8))

sns.scatterplot(X['Income'], X['Score'], hue=X['Labels'], 

                palette=sns.color_palette('hls', 3))

plt.title('KMeans with 3 Clusters')

plt.show()
# Let's see with 5 Clusters

km5 = KMeans(n_clusters=5, random_state = 42).fit(X)



X['Labels'] = km5.labels_

plt.figure(figsize=(12, 8))

sns.scatterplot(X['Income'], X['Score'], hue=X['Labels'], 

                palette=sns.color_palette('hls', 5))

plt.title('KMeans with 5 Clusters')

plt.show()
fig = plt.figure(figsize=(20,8))

ax = fig.add_subplot(121)

sns.swarmplot(x='Labels', y='Income', data=X, ax=ax)

ax.set_title('Labels According to Annual Income')



ax = fig.add_subplot(122)

sns.swarmplot(x='Labels', y='Score', data=X, ax=ax)

ax.set_title('Labels According to Scoring History')



plt.show()
from sklearn.cluster import AgglomerativeClustering 



agglom = AgglomerativeClustering(n_clusters=5).fit(X)



X['Labels'] = agglom.labels_

plt.figure(figsize=(12, 8))

sns.scatterplot(X['Income'], X['Score'], hue=X['Labels'], 

                palette=sns.color_palette('hls', 5))

plt.title('Agglomerative with 5 Clusters')

plt.show()
from scipy.cluster import hierarchy 

from scipy.spatial import distance_matrix 



dist = distance_matrix(X, X)

print(dist)
Z = hierarchy.linkage(dist)
plt.figure(figsize=(18, 50))

dendro = hierarchy.dendrogram(Z, leaf_rotation=0, leaf_font_size=12, orientation='right')
from sklearn.cluster import DBSCAN 



db = DBSCAN(eps=11, min_samples=6).fit(X)



X['Labels'] = db.labels_

plt.figure(figsize=(12, 8))

sns.scatterplot(X['Income'], X['Score'], hue=X['Labels'], 

                palette=sns.color_palette('hls', np.unique(db.labels_).shape[0]))

plt.title('DBSCAN with epsilon 11, min samples 6')

plt.show()

fig = plt.figure(figsize=(20,15))



##### KMeans #####

ax = fig.add_subplot(221)



km5 = KMeans(n_clusters=5).fit(X)

X['Labels'] = km5.labels_

sns.scatterplot(X['Income'], X['Score'], hue=X['Labels'], style=X['Labels'],

                palette=sns.color_palette('hls', 5), s=60, ax=ax)

ax.set_title('KMeans with 5 Clusters')





##### Agglomerative Clustering #####

ax = fig.add_subplot(222)



agglom = AgglomerativeClustering(n_clusters=5).fit(X)

X['Labels'] = agglom.labels_

sns.scatterplot(X['Income'], X['Score'], hue=X['Labels'], style=X['Labels'],

                palette=sns.color_palette('hls', 5), s=60, ax=ax)

ax.set_title('Agglomerative with 5 Clusters')





##### DBSCAN #####

ax = fig.add_subplot(223)



db = DBSCAN(eps=11, min_samples=6).fit(X)

X['Labels'] = db.labels_

sns.scatterplot(X['Income'], X['Score'], hue=X['Labels'], style=X['Labels'], s=60,

                palette=sns.color_palette('hls', np.unique(db.labels_).shape[0]), ax=ax)

ax.set_title('DBSCAN with epsilon 11, min samples 6')



plt.tight_layout()

plt.show()
from sklearn import metrics



print("Silhouette Coefficient Kmeans: %0.3f" % metrics.silhouette_score(X, km5.labels_, metric='sqeuclidean'))

print("Silhouette Coefficient Agglomerative: %0.3f" % metrics.silhouette_score(X, agglom.labels_, metric='sqeuclidean'))

print("Silhouette Coefficient DBSCAN: %0.3f" % metrics.silhouette_score(X, db.labels_, metric='sqeuclidean'))