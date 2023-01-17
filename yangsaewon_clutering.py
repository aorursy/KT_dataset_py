from sklearn.preprocessing import scale

from sklearn.datasets import load_iris

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

%matplotlib inline
iris = load_iris()

iris_df = pd.DataFrame(data=iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

iris_df.head()
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, random_state=2019).fit(iris_df)

iris_df['cluster'] = kmeans.labels_

iris_df['target'] = iris.target

iris_df.head()
iris_result = iris_df.groupby(['target', 'cluster'])['sepal_length'].count()

print(iris_result)
from sklearn.decomposition import PCA



pca = PCA(n_components=2)

pca_transformed = pca.fit_transform(iris.data)



iris_df['pca_x'] = pca_transformed[:, 0]

iris_df['pca_y'] = pca_transformed[:, 1]

iris_df.head()
fig, ax = plt.subplots()

colors = ['red', 'blue', 'green']

for key, group in iris_df.groupby('cluster'):

    group.plot(kind='scatter', x='pca_x', y='pca_y', label=key, ax=ax, color=colors[key])
from sklearn.datasets import make_blobs

from sklearn.cluster import MeanShift

from sklearn.cluster import estimate_bandwidth
# X, y = make_blobs(n_samples=200, n_features=2, centers=3, cluster_std=0.8, random_state=2019)

iris_df = pd.DataFrame(data=iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

iris_df['target'] = iris.target

iris_df.head()
# cluster_df = pd.DataFrame(data=X, columns=['ftr1', 'ftr2'])

# cluster_df['target'] = y
best_bandwidth = estimate_bandwidth(iris.data, quantile=0.2)



meanshift = MeanShift(bandwidth=best_bandwidth)

cluster_label = meanshift.fit_predict(iris.data)

iris_df['cluster'] = cluster_label

print('cluster labels 유형:', np.unique(cluster_label))
iris_df.head()
from sklearn.decomposition import PCA



pca = PCA(n_components=2)

pca_transformed = pca.fit_transform(iris.data)



iris_df['pca_x'] = pca_transformed[:, 0]

iris_df['pca_y'] = pca_transformed[:, 1]

iris_df.head()
fig, ax = plt.subplots()

for key, group in iris_df.groupby('cluster'):

    group.plot(kind='scatter', x='pca_x', y='pca_y', label=key, color=colors[key], ax=ax)
iris_df = pd.DataFrame(data=iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

iris_df['target'] = iris.target

iris_df.head()
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=3, random_state=2019).fit(iris.data)

gmm_cluster_labels = gmm.predict(iris.data)
iris_df['gmm_cluster'] = gmm_cluster_labels
iris_result = iris_df.groupby(['target'])['gmm_cluster'].value_counts()

print(iris_result)
from sklearn.decomposition import PCA



pca = PCA(n_components=2)

pca_transformed = pca.fit_transform(iris.data)



iris_df['pca_x'] = pca_transformed[:, 0]

iris_df['pca_y'] = pca_transformed[:, 1]

iris_df.head()
fig, ax = plt.subplots()

colors = ['red', 'blue', 'green']

for key, group in iris_df.groupby('gmm_cluster'):

    group.plot(kind='scatter', x='pca_x', y='pca_y', label=key, ax=ax, color=colors[key])
iris_df = pd.DataFrame(data=iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

iris_df['target'] = iris.target
from sklearn.cluster import DBSCAN



# eps를 크게하면 반경이 커져 포함하는 데이터가 많아지므로 노이즈 데이터 개수가 작아진다.

# min_samples를 크게 하면 주어진 반경 내에서 더 많은 데이터를 포함시켜야 하므로 노이즈 데이터 개수가 커지게 된다.

dbscan = DBSCAN(eps=0.8, min_samples=8, metric='euclidean')

dbscan_labels = dbscan.fit_predict(iris.data)

iris_df['dbscan_cluster'] = dbscan_labels
iris_result = iris_df.groupby(['target'])['dbscan_cluster'].value_counts()

print(iris_result)
from sklearn.decomposition import PCA



pca = PCA(n_components=2)

pca_transformed = pca.fit_transform(iris.data)



iris_df['pca_x'] = pca_transformed[:, 0]

iris_df['pca_y'] = pca_transformed[:, 1]

iris_df.head()
fig, ax = plt.subplots()

colors = ['red', 'blue', 'green']

for key, group in iris_df.groupby('dbscan_cluster'):

    group.plot(kind='scatter', x='pca_x', y='pca_y', label=key, ax=ax, color=colors[key])