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
data=pd.read_csv("../input/abalone-dataset/abalone.csv")
data.shape
data.describe()
data.head()
data.tail()
data.isnull().sum()
import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import scale

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_samples, silhouette_score

from sklearn.cluster import AgglomerativeClustering



from scipy.cluster.hierarchy import linkage

from scipy.cluster.hierarchy import dendrogram

from scipy.cluster.hierarchy import cut_tree



from sklearn.cluster import DBSCAN

from sklearn import metrics
df = data[['Rings']]
normalized_df=(df-df.mean())/df.std()
ab1 = data.drop('Sex', axis = 1)
ab1 = ab1.drop(['Rings'], 1)
ab1 = pd.concat([ab1,normalized_df],axis=1)
Ks = [2, 3, 4, 5, 6, 7, 8, 9]
ssw = []

for k in Ks:

    kmeans = KMeans(n_clusters = int(k))

    kmeans.fit(ab1)

    sil_score  = silhouette_score(ab1, kmeans.labels_)

    print('silhouette score:',sil_score, 'number of clusters are:', int(k))

    ssw.append(kmeans.inertia_)

plt.plot(Ks, ssw)
k = 3

kmeans = KMeans(n_clusters = k)

kmeans.fit(ab1)
labels = kmeans.labels_

ab1['cluster'] = labels
for i in range(k):

    # select only data observations with cluster label == i

    ds = ab1[ab1['cluster']==i].as_matrix()

    plt.plot(ds[:,0], ds[: , 1], 'o')

plt.show()
kmeans.inertia_
ab1['cluster'].value_counts()