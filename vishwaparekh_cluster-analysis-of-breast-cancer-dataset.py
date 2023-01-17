import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt2

import matplotlib.cm as cm

%matplotlib inline

from sklearn import preprocessing

from subprocess import check_output



#

print(check_output(["ls", "../input"]).decode("utf8"))







# Read the data file

data = pd.read_csv('../input/data.csv')

data.head()

# Cleaning and modifying the data

data = data.drop('id',axis=1)

data = data.drop('Unnamed: 32',axis=1)

# Mapping Benign to 0 and Malignant to 1 

data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})

# Scaling the dataset

datas = pd.DataFrame(preprocessing.scale(data.iloc[:,1:32]))

datas.columns = list(data.iloc[:,1:32].columns)

datas['diagnosis'] = data['diagnosis']

# Creating the high dimensional feature space X

data_drop = datas.drop('diagnosis',axis=1)

X = data_drop.values



#Creating a 2D visualization to visualize the clusters

from sklearn.manifold import TSNE

tsne = TSNE(verbose=1, perplexity=40, n_iter= 4000)

Y = tsne.fit_transform(X)
#Cluster using k-means

from sklearn.cluster import KMeans

kmns = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm='auto')

kY = kmns.fit_predict(X)



f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)



ax1.scatter(Y[:,0],Y[:,1],  c=kY, cmap = "jet", edgecolor = "None", alpha=0.35)

ax1.set_title('k-means clustering plot')



ax2.scatter(Y[:,0],Y[:,1],  c = datas['diagnosis'], cmap = "jet", edgecolor = "None", alpha=0.35)

ax2.set_title('Actual clusters')





#Cluster using Spectral clustering and visualize using Isomap

from sklearn.cluster import SpectralClustering





# Play with gamma to optimize the clustering results

kmns = SpectralClustering(n_clusters=2,  gamma=0.5, affinity='rbf', eigen_tol=0.0, assign_labels='kmeans', degree=3, coef0=1, kernel_params=None, n_jobs=1)

kY = kmns.fit_predict(X)



f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)





ax1.scatter(Y[:,0],Y[:,1],  c=kY, cmap = "jet", edgecolor = "None", alpha=0.35)

ax1.set_title('Spectral clustering plot')



ax2.scatter(Y[:,0],Y[:,1],  c = datas['diagnosis'], cmap = "jet", edgecolor = "None", alpha=0.35)

ax2.set_title('Actual clusters')



# Cluster using hierarchical clustering

from sklearn.cluster import AgglomerativeClustering

aggC = AgglomerativeClustering(n_clusters=2, linkage='ward')

kY = aggC.fit_predict(X)





f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)





ax1.scatter(Y[:,0],Y[:,1],  c=kY, cmap = "jet", edgecolor = "None", alpha=0.35)

ax1.set_title('Hierarchical clustering plot')



ax2.scatter(Y[:,0],Y[:,1],  c = datas['diagnosis'], cmap = "jet", edgecolor = "None", alpha=0.35)

ax2.set_title('Actual clusters')