import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# GaussianMixture

from sklearn.mixture import GaussianMixture



# PCA - use for plot

from sklearn.decomposition import PCA



#ignore warning messages 

import warnings

warnings.filterwarnings('ignore') 
# Get data 

train = pd.read_csv("../input/train.csv")



# Splite data the X - Our data , and y - the prdict label

X = train.drop('label',axis = 1)

y = train['label'].astype('category')



X.head()
dataset = X[y == 6].reset_index().drop("index",axis = 1)



plt.figure(figsize = (10,8))

row, colums = 3, 3

for i in range(9):  

    plt.subplot(colums, row, i+1)

    plt.imshow(dataset.iloc[i].values.reshape(28,28),interpolation='nearest', cmap='Greys')

plt.show()
# Im use PCA to set dematntion into 2D (for plot)

# and use GMM to plot the Gaussian probability distributions



pca = PCA(n_components = 2)

dataset_2D = pca.fit_transform(dataset)



gmm = GaussianMixture(n_components=2)

gmm.fit(dataset_2D)



plt.figure(figsize = (10,8))



# plot GMM center and classes

plt.subplot(2, 1, 1)

plt.scatter(dataset_2D[:,0], dataset_2D[:,1], c = gmm.predict(dataset_2D))

plt.scatter(gmm.means_[:, 0], gmm.means_[:,1])

plt.title("GMM center and classes")



# plot GMM center and score

plt.subplot(2, 1, 2)

plt.scatter(dataset_2D[:,0], dataset_2D[:,1], c = gmm.score_samples(dataset_2D))

plt.scatter(gmm.means_[:, 0], gmm.means_[:,1])

plt.title("GMM center and score")



plt.show()
n_components = [10,20,30,40,50]

aics = []

for n in n_components:

    print(n) # just for progress print 

    model = GaussianMixture(n, covariance_type='full', random_state=0)

    aics.append(model.fit(dataset).aic(dataset))

plt.plot(n_components, aics);
n_components = 20 # selected by AIC



gmm = GaussianMixture(n_components=n_components)

gmm.fit(dataset)
plt.figure(figsize = (10,8))

row, colums = 3, 3

for i in range(9):  

    plt.subplot(colums, row, i+1)

    toShow = gmm.sample()[0]

    plt.imshow(toShow.reshape(28,28),interpolation='nearest', cmap='Greys')

plt.show()