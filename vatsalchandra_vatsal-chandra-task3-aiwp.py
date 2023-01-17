# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
K = 15

col_names = ["customer_id","gender","age","annual_income","spending_score"]

data = pd.read_csv("../input/mall-customers/Mall_Customers.csv",names=col_names,header=0)

data = data.sample(frac=1)

data.head()
import seaborn as sns

import matplotlib.pyplot as plt
sns.pairplot(data)
print(data['gender'].value_counts())

sns.countplot(x=data['gender'])

plt.show
sns.distplot(data['age'],bins=50)

plt.show()

sns.distplot(data['spending_score'],bins=50)

plt.show()
plt.figure(figsize=(15,6))

sns.countplot(x=data['age'])

plt.show()
plt.figure(figsize=(20,6))

sns.countplot(x=data['annual_income'])

plt.show()
data = data.iloc[:,[1,2,4]]

data = data.replace(to_replace="Female",value=1)

data = data.replace(to_replace="Male",value=0)

data.head()
fig = plt.figure(figsize=(14,8))

ax = plt.axes(projection="3d")



x = data["age"].values

z = data["gender"].values

y = data["spending_score"].values



img = ax.scatter(x, y, z, s=50, edgecolors='r')

ax.set_zticks([0, 1])

ax.set_xlabel('Age')

ax.set_ylabel('Spending Score')

ax.set_zlabel('Gender (F = 1, M = 0)')

plt.show()
X = data.iloc[:, 1:].values

X[:10,:]
from sklearn.cluster import KMeans

wcss = []

for i in range(1, 11):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)

    kmeans.fit(X)

    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)

plt.title('Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('Within Cluster Sum of Squares)')

plt.show()
# Fitting K-Means to the dataset.

kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 0)

pred = kmeans.fit_predict(X)

# Visualising the clusters

plt.figure(figsize=(15,10))

plt.scatter(X[pred == 0, 0], X[pred == 0, 1], s = 100, c = 'red')

plt.scatter(X[pred == 1, 0], X[pred == 1, 1], s = 100, c = 'blue')

plt.scatter(X[pred == 2, 0], X[pred == 2, 1], s = 100, c = 'green')

plt.scatter(X[pred == 3, 0], X[pred == 3, 1], s = 100, c = 'cyan')



plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'black', label = 'Centroids')

plt.title('Clusters of customers')

plt.xlabel('Annual Income (k$)')

plt.ylabel('Spending Score (1-100)')

plt.legend()

plt.show()
vals = data.values

k = 4  #elbow method



kmeans = KMeans(n_clusters = k, init = 'k-means++').fit(vals)



fig = plt.figure(figsize=(14,8))

ax = plt.axes(projection="3d")



x = data["age"].values

z = data["gender"].values

y = data["spending_score"].values



print(f"K-means: num of clusters - {k}")



ax.scatter3D(x, y, z, s=100, c=kmeans.labels_, edgecolors='r')

ax.scatter3D(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2],kmeans.cluster_centers_[:, 0], s = 300,

             color = 'black', marker="P")



# fig.colorbar(img1)

ax.set_zticks([0, 1])

ax.set_xlabel('Age')

ax.set_ylabel('Spending Score')

ax.set_zlabel('Gender (F = 1, M = 0)')

plt.show()
from sklearn.cluster import MeanShift, estimate_bandwidth



vals = data.values

bandwidth = estimate_bandwidth(vals, quantile=0.2, n_samples=20)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(vals)

cluster_centers = ms.cluster_centers_

num_clusters = len(cluster_centers)



print(f"Mean-shift: num of clusters - {num_clusters}")



fig = plt.figure(figsize=(14,8))

ax = plt.axes(projection="3d")



x = data["age"].values

z = data["gender"].values

y = data["spending_score"].values



ax.scatter3D(x, y, z, s=100, c=ms.labels_, edgecolors='r')

ax.scatter3D(ms.cluster_centers_[:, 1], ms.cluster_centers_[:, 2],ms.cluster_centers_[:, 0], s = 300,

             color = 'black', marker="P")



# fig.colorbar(img1)

ax.set_zticks([0, 1])

ax.set_xlabel('Age')

ax.set_ylabel('Spending Score')

ax.set_zlabel('Gender (F = 1, M = 0)')

plt.show()
from sklearn.cluster import AffinityPropagation



afprop = AffinityPropagation().fit(vals)

cluster_centers = afprop.cluster_centers_indices_

num_clusters = len(cluster_centers)

print(f"Affinity Propagation: num of clusters - {num_clusters}")



fig = plt.figure(figsize=(14,8))

ax = plt.axes(projection="3d")



x = data["age"].values

z = data["gender"].values

y = data["spending_score"].values



ax.scatter3D(x, y, z, s=100, c=afprop.labels_, edgecolors='r')

ax.scatter3D(afprop.cluster_centers_[:, 1], afprop.cluster_centers_[:, 2],afprop.cluster_centers_[:, 0], s = 300, 

             color = 'black', marker="P")



# fig.colorbar(img1)

ax.set_zticks([0, 1])

ax.set_xlabel('Age')

ax.set_ylabel('Spending Score')

ax.set_zlabel('Gender (F = 1, M = 0)')

plt.show()