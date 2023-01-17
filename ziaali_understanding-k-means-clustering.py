# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



data=pd.read_csv("../input/Iris.csv")

data.head(3)



# Any results you write to the current directory are saved as output.
iris=data.drop(["Species"],axis=1)



plt.scatter(iris.iloc[:, 0], iris.iloc[:, 1], s=50);


sse = []



for i in range(1, 15):

    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=500, n_init=5, random_state=0)#doing clustering

    kmeans.fit(iris)

    sse.append(kmeans.inertia_)



print('Models created')

# Plotting the results onto a line graph, allowing us to observe 'The elbow curve'

plt.plot(range(1, 15), sse)

plt.title('The elbow method')

plt.xlabel('Number of clusters "k"')

plt.ylabel('SSE')  #  sum of squares errors

plt.show()

## Creating model

kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 1000, n_init = 10, random_state = 0).fit(iris)

# y_kmeans = kmeans.fit_predict(data)/



print(kmeans.labels_)



data['Cluster'] = kmeans.labels_
y_kmeans = kmeans.predict(iris)

#Define our own color map



print(kmeans.labels_)

import matplotlib.pyplot as plt



plt.scatter(iris.iloc[:,0],iris.iloc[:,1], c=y_kmeans) 

centers = kmeans.cluster_centers_

plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);

plt.show()