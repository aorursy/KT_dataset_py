import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
df=pd.read_excel("../input/CNumerical.xls", "Sheet1")
print(df.head(10))
plt.rcParams['figure.figsize']=(16, 9)
plt.style.use('ggplot')

#getting the values and plotting it
bp = df['blood pressure'].values
chol = df['cholesterol'].values
plotting = np.array(list(zip(bp, chol)))
plt.scatter(bp, chol, c='black', s=7)
#Euclidean Distance Calculator
def dist(a, b, ax=1):
    return np.linalg.norm(a-b, axis=ax)

#Number of clusters
k=3
# X Coordinates of random centroids
C_x = np.random.randint(0, np.max(plotting) - 20, size=k)
# Y Coordinates of random centroids
C_y = np.random.randint(0, np.max(plotting) - 20, size=k)

C = np.array(list(zip(C_x, C_y)), dtype=np.float32)

print(C)
#plotting along with the Centroids
plt.scatter(bp, chol, c='#050505', s=7)
plt.scatter(C_x, C_y, marker='*', s=200, c='g')

#To store the value of centroids when it updates
C_old = np.zeros(C.shape)

#Cluster Lables(0, 1, 2)
clusters = np.zeros(len(plotting))

#Error func. - Distance between new centroids and old centroids
error = dist(C, C_old, None)
#Loop will run till the error becomes zero
while error != 0:
    #assigning each value to its cloest cluster
    for i in range(len(plotting)):
        distances = dist(plotting[i], C)
        cluster = np.argmin(distances)
        cluster[i] = cluster
        
    #storing  the old centroid values
    C_old = deepcopy(C)
    
    #finding the new centriods by taking the average value
    for i in range(k):
        points = [plotting[j] for j in range(len(plotting)) if clusters[j] == i]
        C[i] = np.mean(points, axis=0)

error = dist(C, C_old, None)

colors = ['r','g', 'b', 'y', 'c', 'm']
fig, ax = plt.subplots()
for i in range(k):
    points = np.array([plotting[j] for j in range (len(plotting)) if clusters[j] == i])
    ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#050505')
#numbers of clusters           
kmeans = Kmeans(n_clusters=3)
#fitting the input data
kmeans = kmeans.fit(plotting)
#getting the cluster labels
labels = kmeans.predict(plotting)
#centroid values
centroids=kmeans.cluster_centers_

print(C)

print(centroids) 