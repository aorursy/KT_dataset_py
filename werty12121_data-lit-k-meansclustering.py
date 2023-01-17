# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

sns.set_palette("dark")

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split

from sklearn.manifold import TSNE

import random



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from scipy.io import loadmat

mnist = loadmat("../input/mnist-original.mat")

mnist_data = mnist["data"].T

mnist_label = mnist["label"][0]
data=pd.DataFrame(mnist_data)

data.head()
label=pd.DataFrame(mnist_label)



label[0]=label[0].apply(int)

label.info()
data.isnull().sum().sum(),label.isnull().sum().sum()
X_train,X_test,y_train,y_test=train_test_split(data,label,shuffle=True,test_size=0.2)
y_train.head()
tsne=TSNE(2,perplexity=40,n_iter=1200,early_exaggeration=30,verbose=1)

# tsne.fit(data[:500])

transformed=tsne.fit_transform(X_train[:7000])

transformed.shape
f, ax = plt.subplots(figsize=(20, 20))

cmap = sns.hls_palette(10, l=.3, s=.8)

filled_markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', 'h', 'H', 'D', 'd', 'P', 'X','*')



sns.scatterplot(transformed[:,0],transformed[:,1],hue=y_train[:7000].values.T[0],palette=cmap)
# from sklearn.neighbors import KNeighborsClassifier

# model=KNeighborsClassifier(9,'distance')

# model.fit(transformed[:15000],y_train[:15000].values.T[0])

# model.score(transformed[15000:],y_train[15000:18000].values.T[0])
class OwnKMeans():

    

    def __init__(self,k=5):

        self.k=k

    

    def fit(self,X,n_iter=1000):

        self.centroids=X[np.random.randint(0,X.shape[0],(self.k))]

        for iterat in range(n_iter):

            classifications={}

            for k in range(self.k):

                classifications[k]=[]

            for x in X:

                dist=[np.linalg.norm(x-centroid) for centroid in self.centroids]

                classification=np.argmin(dist,axis=0)

                classifications[classification].append(x)



            for ci,c in enumerate(self.centroids):

                self.centroids[ci]=np.average(classifications[ci],axis=0)

            

    

    def predict(self,X):

        return np.argmin([[np.linalg.norm(x-centroid) for x in X ]for centroid in self.centroids],axis=0)
mod1=OwnKMeans(10)

mod1.fit(transformed[:5000])

preds=mod1.predict(transformed[:7000])



f, ax = plt.subplots(figsize=(20, 20))

sns.scatterplot(transformed[:,0],transformed[:,1],hue=preds[:7000],palette=cmap)
km=KMeans(10)

km.fit(transformed[:7000])

predictions=km.predict(transformed[:7000])

print(predictions)



f, ax = plt.subplots(figsize=(20, 20))

sns.scatterplot(transformed[:,0],transformed[:,1],hue=predictions[:7000],palette=cmap)