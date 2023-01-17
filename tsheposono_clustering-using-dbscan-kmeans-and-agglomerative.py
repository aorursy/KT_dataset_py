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
import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline 

import seaborn as sns

sns.set("talk","darkgrid",font_scale=1,font="sans-serif",color_codes=True)

from sklearn import metrics

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

from sklearn.cluster import DBSCAN

from sklearn.cluster import AgglomerativeClustering

from mpl_toolkits.mplot3d import Axes3D
df = pd.read_csv(r"../input/xclara/xclara.csv")

df.head()
X = df[["V1"]]

Y = df[["V2"]]
Nc = range(1,20)

kmeans = [KMeans(n_clusters=i) for i in Nc]

kmeans

score = [kmeans[i].fit(Y).score(Y) for i in range(len(kmeans))]

score

fig, ax = plt.subplots(figsize=(10,10))

plt.plot(Nc,score)

plt.title("Elbow Curve")

plt.ylabel("Scores")

plt.xlabel("Number of clustering")

plt.show()
pca = PCA(n_components=1).fit(Y)

pca_d = pca.transform(X)

pca_c = pca.transform(Y)
kmeans = KMeans(n_clusters=3)

kmeans_output = kmeans.fit(Y)
kmeans_output.labels_
np.unique(kmeans_output.labels_)
fig = plt.figure(figsize=(10,10))

ax = Axes3D(fig)

ax.scatter(pca_c[:,0],pca_d[:,0],c=kmeans_output.labels_, s=3, cmap="viridis")

plt.title("KMeans clustering")

plt.xlabel("V1")

plt.ylabel("V2")

plt.show()
dbscan = DBSCAN

dbscan_output = kmeans.fit(Y)
dbscan_output.labels_
np.unique(dbscan_output.labels_)
fig = plt.figure(figsize=(10,10))

ax = Axes3D(fig)

ax.scatter(pca_c[:,0],pca_d[:,0],c=dbscan_output.labels_, s=3, cmap="viridis")

plt.title("DBSCAN clustering")

plt.xlabel("V1")

plt.ylabel("V2")

plt.show()
agglo = AgglomerativeClustering(n_clusters=3)

agglo_output = agglo.fit(Y)
agglo_output.labels_
np.unique(agglo_output.labels_)
fig = plt.figure(figsize=(10,10))

ax = Axes3D(fig)

ax.scatter(pca_c[:,0],pca_d[:,0],c=agglo_output.labels_, s=3, cmap="viridis")

plt.title("Agglomerative Clusting")

plt.xlabel("V1")

plt.ylabel("V2")

plt.show()