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
data=pd.read_csv('../input/winequalityred/winequality-red.csv')
data.head()
data.tail()
wine=data[["sulphates","alcohol"]]
from sklearn.preprocessing import scale

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_samples,silhouette_score
wine_std=pd.DataFrame(scale(wine),columns=list(wine.columns))
import matplotlib.pyplot as plt

import seaborn as sns
Ks=[2,3,4,5,6,7,8,9]
ssw=[]

for k in Ks:

    kmeans=KMeans(n_clusters=int(k))

    kmeans.fit(wine_std)

    sil_score=silhouette_score(wine_std,kmeans.labels_)

    print("silhouette_samples",sil_score,"number of clusters are:",int(k))

    ssw.append(kmeans.inertia_)

plt.plot(Ks,ssw)

    
k=3

kmeans=KMeans(n_clusters=k)

kmeans.fit(wine_std)
labels=kmeans.labels_

wine_std["cluster"]=labels

for i in range(k):

    ds=wine_std[wine_std["cluster"]==i].as_matrix()

    plt.plot(ds[:,0],ds[:,1],'o')

plt.plot()
from sklearn.cluster import AgglomerativeClustering
for n_clusters in range(2,10):

    cluster_model=AgglomerativeClustering(n_clusters=n_clusters,affinity='euclidean',linkage='ward')

    cluster_labels=cluster_model.fit_predict(wine_std)

    silhouette_avg=silhouette_score(wine_std,cluster_labels,metric='euclidean')

    

plt.plot()
s=3

hclust=AgglomerativeClustering(n_clusters=s,affinity='euclidean',linkage='ward')

hclust.fit(wine_std)
for i in range(s):

    hc=wine_std[wine_std["cluster"]==i].as_matrix()

    plt.plot(hc[:,0],hc[:,1],'o')

plt.show()