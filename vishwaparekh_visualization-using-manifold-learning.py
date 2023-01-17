import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import matplotlib as mpl

import matplotlib.pyplot as plt

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
# Visualization using tSNE

from sklearn.manifold import TSNE

tsne = TSNE(verbose=1, perplexity=40, n_iter= 4000)

tsne = tsne.fit_transform(X)

plt.scatter(tsne[:,0],tsne[:,1],  c = datas['diagnosis'], cmap = "jet", edgecolor = "None", alpha=0.35)

plt.title('t-SNE Scatter Plot')
# Visualization using Isomap

from sklearn.manifold import Isomap

Isomp = Isomap(n_neighbors=150, n_components=2, eigen_solver='auto', tol=0, max_iter=None, path_method='auto', neighbors_algorithm='auto', n_jobs=1)



Y = Isomp.fit_transform(X)

plt.scatter(Y[:,0],Y[:,1],  c = datas['diagnosis'], cmap = "jet", edgecolor = "None", alpha=0.35)

plt.title('Isomap Scatter Plot')
# Visualization using LLE

from sklearn.manifold import LocallyLinearEmbedding

LLE_Train = LocallyLinearEmbedding(n_neighbors=5, n_components=2, method='standard')



Y = LLE_Train.fit_transform(X)

plt.scatter(Y[:,0],Y[:,1],  c = datas['diagnosis'], cmap = "jet", edgecolor = "None", alpha=0.35)

plt.title('LLE Scatter Plot')