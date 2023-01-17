%matplotlib inline

from time import time

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn import metrics

from sklearn.cluster import KMeans

from sklearn.datasets import load_digits

from sklearn.decomposition import PCA

from sklearn.preprocessing import scale



np.random.seed(5)
def load_data(path):

    with np.load(path) as f:

        x_train, y_train = f['x_train'], f['y_train']

        x_test, y_test = f['x_test'], f['y_test']

        return (x_train, y_train), (x_test, y_test)



(X_train, y_train), (X_test, y_test) = load_data('../input/mnistnpz/mnist.npz')
import matplotlib.image as mpimg

import matplotlib.cm as cm



fig, ax = plt.subplots(ncols=10, nrows=1, figsize=(10, 5))

amostra = np.random.choice(len(X_train), 10) #escolhe 10 imagens dentre as 60000



for i in range(len(amostra)):

    imagem = np.array(X_train[amostra[i]])

    ax[i].imshow(imagem, cmap = cm.Greys_r)

    ax[i].get_xaxis().set_ticks([])

    ax[i].get_yaxis().set_ticks([])

    ax[i].set_title(y_train[amostra[i]]) # Coloca o label como título da figura.

plt.show()
print(X_train.shape)

X_train = X_train.astype('float32').reshape(X_train.shape[0], X_train.shape[1]* X_train.shape[2])

X_train = X_train/255. #escala o dado entre 0 e 1

print(X_train.shape)
amostra = np.random.choice(X_train.shape[0], 10000)

X_train = X_train[amostra]

y_train = y_train[amostra]
%%time

kmeans = KMeans(init='random', n_clusters=10, n_init=50).fit(X_train)
c0 = np.where(kmeans.labels_==0)[0]

print(y_train[c0[:30]])
pd.crosstab(kmeans.labels_, y_train,rownames=['cluster'], colnames=['número'])
def imprime_dez_exemplos_cluster(cluster):

    fig, ax = plt.subplots(ncols=10, nrows=1, figsize=(10, 5))

    amostra = np.random.choice(len(cluster), 10) #escolhe 10 imagens dentre as 60000



    for i in range(len(amostra)):

        imagem = np.array(X_train[cluster[amostra[i]]]).reshape(28,28)

        ax[i].imshow(imagem, cmap = cm.Greys_r)

        ax[i].get_xaxis().set_ticks([])

        ax[i].get_yaxis().set_ticks([])

        ax[i].set_title(y_train[cluster[amostra[i]]]) # Coloca o label como título da figura.

    plt.show()
imprime_dez_exemplos_cluster(c0)
for i in range(10):

    c = np.where(kmeans.labels_==i)[0]

    print('Cluster ', i)

    imprime_dez_exemplos_cluster(c)