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
#LOAD DATASET AND ALL MODULES

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.pyplot as plt

data = pd.read_csv("/kaggle/input/german-credit/german_credit_data.csv")
data.head()
percentage_missing_value = (data.isnull().sum())/(len(data)) * 100

percentage_missing_value
categorical_missing = ['Saving accounts', 'Checking account']

for x in categorical_missing :

    data[x] = data[x].fillna(data[x].mode().values[0])
data = data.drop(columns = ['Unnamed: 0'])
data.info()
data.head()

data['Job'] = data['Job']+1 #menghindari infinity value ketika transformasi data
data_numerik = ['Age', 'Credit amount', 'Duration', 'Job']

data_kategorik = ['Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']

data.head()
data.describe()
fig = plt.figure(figsize = (20,20))

axes = 520

for cat in data_kategorik:

    axes += 1

    fig.add_subplot(axes)

    sns.countplot(data = data, x = cat, hue ='Purpose')

plt.show()
fig = plt.figure(figsize = (20,20))

axes = 520

for cat in data_kategorik:

    axes += 1

    fig.add_subplot(axes)

    sns.barplot(data = data, x = cat, y ='Duration')

plt.show()
data_cluster = data[['Age', 'Credit amount', 'Duration']]

data.head()
import seaborn as sns

cor = data_cluster.corr() #Calculate the correlation of the above variables

plt.figure(figsize=(10,10))

sns.heatmap(cor, square = True) #Plot the correlation as heat map
plt.figure(figsize=(15,3))

sns.boxplot(x = data_cluster['Age'])

plt.show()

plt.figure(figsize=(15,3))

sns.boxplot(x = data_cluster['Credit amount'])

plt.show()

plt.figure(figsize=(15,3))

sns.boxplot(x = data_cluster['Duration'])

plt.show()
%matplotlib inline 

import matplotlib.pyplot as plt 

import numpy as np 

import pandas as pd 

from sklearn.cluster import KMeans
#Data diberlakukan tranformasi logaritmik agar variansi data lebih masuk kedalam range

cluster_credit_duration = np.log(data_cluster[['Age','Credit amount', 'Duration']])

cluster_credit_duration.head()
X = np.array(cluster_credit_duration)
#Scree Plot untuk menentukan nilai K

Sum_of_squared_distances = []

K = range(1,15)

for k in K:

    km = KMeans(n_clusters=k)

    km = km.fit(X)

    Sum_of_squared_distances.append(km.inertia_)

plt.figure(figsize = (15,5))

plt.plot(K, Sum_of_squared_distances, 'bx-')

plt.xlabel('k')

plt.ylabel('Sum_of_squared_distances')

plt.title('Elbow Method For Optimal k')

plt.show()
#mengaktifkan algoritma k-means

kmeans = KMeans(n_clusters=3)

kmeans.fit(X)
print(kmeans.cluster_centers_)
print(np.exp(kmeans.cluster_centers_))
from mpl_toolkits.mplot3d import Axes3D



fig = plt.figure(num = None, figsize = (10,5), dpi = 80, facecolor = 'w', edgecolor ='k')

ax = Axes3D(fig)

ax.scatter3D(X[:,0], X[:,1], X[:,2], c = kmeans.labels_, cmap = 'rainbow')

ax.scatter3D(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], kmeans.cluster_centers_[:,2], c = 'black')



xLabel = ax.set_xlabel('Age', linespacing = 3.2)

yLabel = ax.set_ylabel('Credit amount', linespacing = 1.2)

zLabel = ax.set_zlabel('Duration', linespacing = 1.5)
print(kmeans.labels_)
data['Cluster (K-Means)'] = kmeans.labels_
data.head()
#arr = []

#for i in range(len(data['Cluster (K-Means)'])):

#    if data['Cluster (K-Means)'][i] == 0:

#        arr.append('Good Credit')

#    elif data['Cluster (K-Means)'][i] == 1:

#        arr.append('Medium Credit')

#    else :

#        arr.append('Bad Credit')

#data['Cluster (K-Means)']=arr
data.head()
#Membuat feature pembayaran kredit per waktu, dibuat untuk memvalidasi cluster. 

# Cluster 0 adalah cluster good credit yang berarti jumlah kredit banyak namun durasi pembayaran sebentar.

data['Pay/Time'] = data['Credit amount'] / data['Duration']
data.sort_values(by = ['Pay/Time'], ascending = False).head(10)
data.sort_values(by = ['Pay/Time'], ascending = True).head()
plt.figure(figsize=(15,5))

sns.countplot(data=data, x='Cluster (K-Means)', hue='Purpose')
fig = plt.figure(figsize = (20,20))

axes = 520

for cat in data_kategorik:

    axes += 1

    fig.add_subplot(axes)

    sns.countplot(data = data, y = cat, hue ='Cluster (K-Means)')

plt.show()
from sklearn.cluster import AgglomerativeClustering

import scipy.cluster.hierarchy as sch

model = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')

model.fit(X)

labels = model.labels_

plt.figure(figsize = (20,25))

dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import axes3d

fig = plt.figure(num = None, figsize = (10,5), dpi = 80, facecolor = 'w', edgecolor ='k')

#ax = plt.axes(prohection='3d')

ax = Axes3D(fig)

ax.scatter3D(X[:,0], X[:,1], X[:,2], c = labels, cmap = 'rainbow')

#ax.scatter3D(labels.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], kmeans.cluster_centers_[:,2], c = 'black')



xLabel = ax.set_xlabel('Age', linespacing = 3.2)

yLabel = ax.set_ylabel('Credit amount', linespacing = 1.2)

zLabel = ax.set_zlabel('Duration', linespacing = 1.5)
print(labels)
data['Cluster (Hierarchical)'] = labels
data.head(10)
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps = 0.09, min_samples = 3)

dbscan.fit(X)
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import axes3d

fig = plt.figure(num = None, figsize = (15,10), dpi = 80, facecolor = 'w', edgecolor ='k')

#ax = plt.axes(prohection='3d')

ax = Axes3D(fig)

ax.scatter3D(X[:,0], X[:,1], X[:,2], c = dbscan.labels_, cmap = 'rainbow')

#ax.scatter3D(labels.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], kmeans.cluster_centers_[:,2], c = 'black')





xLabel = ax.set_xlabel('Age', linespacing = 3.2)

yLabel = ax.set_ylabel('Credit amount', linespacing = 1.2)

zLabel = ax.set_zlabel('Duration', linespacing = 1.5)
print(dbscan.labels_)
####
data