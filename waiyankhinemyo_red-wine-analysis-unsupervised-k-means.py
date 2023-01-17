#import libraries 

#structures
import numpy as np
import pandas as pd

#visualization
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()
from mpl_toolkits.mplot3d import Axes3D

#get model duration
import time
from datetime import date

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#load dataset
data = '../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv'
dataset = pd.read_csv(data)
dataset.shape
dataset.dtypes
dataset.describe()
#check for missing data
dataset.isnull().any().any()
#check for unreasonable data
dataset.applymap(np.isreal)
sns_plot = sns.pairplot(dataset)
sns_plot = sns.distplot(dataset['quality'])
#set x and y
from sklearn.preprocessing import StandardScaler

X = dataset.iloc[:,0:11]
y = dataset['quality']

#stadardize data
X_scaled = StandardScaler().fit_transform(X)
dataset.head()
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
pca = PCA(n_components=6)
pc_X = pca.fit_transform(X_scaled)
pc_columns = ['pc1','pc2','pc3','pc4','pc5','pc6']
print(pca.explained_variance_ratio_.sum())
print(pca.explained_variance_ratio_)
#get correlation map
corr_mat=dataset.corr()
#visualise data
plt.figure(figsize=(13,5))
sns_plot=sns.heatmap(data=corr_mat, annot=True, cmap='GnBu')
plt.show()
#check for highly correlated values to be removed
target = 'quality'
candidates = corr_mat.index[
    (corr_mat[target] > 0.5) | (corr_mat[target] < -0.5)
].values
candidates = candidates[candidates != target]
print('Correlated to', target, ': ', candidates)
#import libraries
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
#try to find optimal k using the elbow method
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300, n_init=12, random_state=0)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
f3, ax = plt.subplots(figsize=(8, 6))
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
#Applying kmeans to the dataset, set k=2
kmeans = KMeans(n_clusters = 2)
start_time = time.time()
clusters = kmeans.fit_predict(X_scaled)
today = date.today()
print("--- %s seconds ---" % (time.time() - start_time))
labels = kmeans.labels_
#2D plot
colors = 'rgbkcmy'
for i in np.unique(clusters):
    plt.scatter(X_scaled[clusters==i,0],
               X_scaled[clusters==i,1],
               color=colors[i], label='Cluster' + str(i+1))
plt.legend()
# Visualise the clusterds considerig fixed acidity, residual sugar, and alcohol
fig = plt.figure(figsize=(20, 15))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=15, azim=40)

ax.scatter(X_scaled[:,0], X_scaled[:,3], X_scaled[:,10],c=y, edgecolor='k')
ax.set_xlabel('Acidity')
ax.set_ylabel('Sugar')
ax.set_zlabel('Alcohol')
ax.set_title('K=2: Acidity, Sugar, Alcohol', size=22)
#evaluate model
from sklearn import metrics
from sklearn.metrics import pairwise_distances
metrics.silhouette_score(X_scaled, labels, metric='euclidean')
kmeans.inertia_
#Applying kmeans to the dataset, set k=2
kmeans = KMeans(n_clusters = 2)
start_time = time.time()
clusters = kmeans.fit_predict(pc_X)
today = date.today()
print("--- %s seconds ---" % (time.time() - start_time))
labels = kmeans.labels_
#2D plot
colors = 'rgbkcmy'
for i in np.unique(clusters):
    plt.scatter(pc_X[clusters==i,0],
               pc_X[clusters==i,1],
               color=colors[i], label='Cluster' + str(i+1))
plt.legend()
#evaluate model
metrics.silhouette_score(pc_X, labels, metric='euclidean')
kmeans.inertia_