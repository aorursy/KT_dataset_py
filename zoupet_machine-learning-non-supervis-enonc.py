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



## numpy

## pandas 

## matplotlib (pyplot)

## seaborn



%matplotlib inline 



import os

import warnings



warnings.filterwarnings('ignore')
df.rename(index=str, columns=____________, inplace=True)

df.head()
sns.____________(df.drop('CustomerID', axis=1), hue='____________', aspect=1.5)

plt.show()
X = df.drop(['____________', '____________'], axis=1)
## Kmeans à importer



clusters = []



for i in range(1, ____________):

    km = KMeans(____________).fit(X)

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

km3 = ____________



X['Labels'] = ____________

plt.figure(figsize=(12, 8))

sns.scatterplot(X['____________'], X['____________'], hue=X['____________'], 

                palette=sns.color_palette('hls', 3))

plt.title('KMeans with 3 Clusters')

plt.show()
# Let's see with 5 Clusters

km5 = ____________



X['Labels'] = ____________

plt.figure(figsize=(12, 8))

sns.scatterplot(X['____________'], X['____________'], hue=X['____________'], 

                palette=sns.color_palette('hls', 5))

plt.title('KMeans with 5 Clusters')

plt.show()
fig = plt.figure(figsize=(20,8))

ax = fig.add_subplot(121)

sns.____________(x='____________', y='____________', data=X, ax=ax)

ax.set_title('Labels According to Annual Income')



ax = fig.add_subplot(122)

sns.____________(x='____________', y='____________', data=X, ax=ax)

ax.set_title('Labels According to Scoring History')



plt.show()
## Importer AgglomerativeClustering 



agglom = ____________



X['Labels'] = ____________

plt.figure(figsize=(12, 8))

sns.scatterplot(X['____________'], X['____________'], hue=X['____________'], 

                palette=sns.color_palette('hls', 5))

plt.title('Agglomerative with 5 Clusters')

plt.show()
from scipy.cluster import hierarchy 

from scipy.spatial import distance_matrix 



dist = ____________

print(dist)
Z = hierarchy.linkage(____________)
plt.figure(figsize=(18, 50))

dendro = hierarchy.dendrogram(____________, leaf_rotation=0, leaf_font_size=12, orientation='right')
# Importer DBSCAN 



db = ____________



X['Labels'] = ____________

plt.figure(figsize=(12, 8))

sns.scatterplot(X['____________'], X['____________'], hue=X['____________'], 

                palette=sns.color_palette('hls', np.unique(db.labels_).shape[0]))

plt.title('DBSCAN with epsilon 11, min samples 6')

plt.show()

fig = plt.figure(figsize=(20,15))



##### KMeans #####

ax = fig.add_subplot(221)



km5 = ____________

X['Labels'] = ____________

sns.scatterplot(X['Income'], X['____________'], hue=X['____________'], style=X['____________'],

                palette=sns.color_palette('hls', 5), s=60, ax=ax)

ax.set_title('KMeans with 5 Clusters')





##### Agglomerative Clustering #####

ax = fig.add_subplot(222)



agglom = ____________

X['Labels'] = ____________

sns.scatterplot(X['Income'], X['____________'], hue=X['____________'], style=X['____________'],

                palette=sns.color_palette('hls', 5), s=60, ax=ax)

ax.set_title('Agglomerative with 5 Clusters')





##### DBSCAN #####

ax = fig.add_subplot(223)



db = ____________

X['Labels'] = ____________

sns.scatterplot(X['____________'], X['____________'], hue=X['____________'], style=X['____________'], s=60,

                palette=sns.color_palette('hls', np.unique(db.labels_).shape[0]), ax=ax)

ax.set_title('DBSCAN with epsilon 11, min samples 6')



plt.tight_layout()

plt.show()
from sklearn import metrics



print("Silhouette Coefficient Kmeans: %0.3f" % ____________(X, ____________, metric='sqeuclidean'))

print("Silhouette Coefficient Agglomerative: %0.3f" % ____________(X, ____________, metric='sqeuclidean'))

print("Silhouette Coefficient DBSCAN: %0.3f" % ____________(X, ____________, metric='sqeuclidean'))