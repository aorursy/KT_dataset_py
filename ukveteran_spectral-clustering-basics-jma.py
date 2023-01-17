import pandas as pd

import numpy as np



data = pd.read_csv('../input/chameleon/chameleon.data', delimiter=' ', names=['x','y'])

data.plot.scatter(x='x',y='y')
from sklearn.cluster import DBSCAN



db = DBSCAN(eps=15.5, min_samples=5).fit(data)

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)

core_samples_mask[db.core_sample_indices_] = True

labels = pd.DataFrame(db.labels_,columns=['Cluster ID'])

result = pd.concat((data,labels), axis=1)

result.plot.scatter(x='x',y='y',c='Cluster ID', colormap='jet')
data1 = pd.read_csv('../input/2d-data/2d_data.txt', delimiter=' ', names=['x','y'])

data2 = pd.read_csv('../input/elliptical/elliptical.txt', delimiter=' ', names=['x','y'])



fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,5))

data1.plot.scatter(x='x',y='y',ax=ax1)

data2.plot.scatter(x='x',y='y',ax=ax2)
from sklearn import cluster



k_means = cluster.KMeans(n_clusters=2, max_iter=50, random_state=1)

k_means.fit(data1)

labels1 = pd.DataFrame(k_means.labels_,columns=['Cluster ID'])

result1 = pd.concat((data1,labels1), axis=1)



k_means2 = cluster.KMeans(n_clusters=2, max_iter=50, random_state=1)

k_means2.fit(data2)

labels2 = pd.DataFrame(k_means2.labels_,columns=['Cluster ID'])

result2 = pd.concat((data2,labels2), axis=1)



fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,5))

result1.plot.scatter(x='x',y='y',c='Cluster ID',colormap='jet',ax=ax1)

ax1.set_title('K-means Clustering')

result2.plot.scatter(x='x',y='y',c='Cluster ID',colormap='jet',ax=ax2)

ax2.set_title('K-means Clustering')
from sklearn import cluster

import pandas as pd



spectral = cluster.SpectralClustering(n_clusters=2,random_state=1,affinity='rbf',gamma=5000)

spectral.fit(data1)

labels1 = pd.DataFrame(spectral.labels_,columns=['Cluster ID'])

result1 = pd.concat((data1,labels1), axis=1)



spectral2 = cluster.SpectralClustering(n_clusters=2,random_state=1,affinity='rbf',gamma=100)

spectral2.fit(data2)

labels2 = pd.DataFrame(spectral2.labels_,columns=['Cluster ID'])

result2 = pd.concat((data2,labels2), axis=1)



fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,5))

result1.plot.scatter(x='x',y='y',c='Cluster ID',colormap='jet',ax=ax1)

ax1.set_title('Spectral Clustering')

result2.plot.scatter(x='x',y='y',c='Cluster ID',colormap='jet',ax=ax2)

ax2.set_title('Spectral Clustering')