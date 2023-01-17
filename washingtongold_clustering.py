from sklearn.datasets.samples_generator import make_blobs

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns
x, y = make_blobs(n_samples=1000, centers=3, n_features=2,random_state = 73)

x1 = []

x2 = []

for i in range(len(x)):

    x1.append(x[i][0])

    x2.append(x[i][1])

fig = plt.figure(figsize=(8,8))

sns.scatterplot(x1,x2,hue=y,palette = 'icefire')
fig = plt.figure(figsize=(8,8))

sns.scatterplot(x1,x2)
import pandas as pd

data = pd.DataFrame({'x':x1,'y':x2})

data.head()
from sklearn.cluster import KMeans

cluster = KMeans(n_clusters = 3, n_jobs=-1)

model = cluster.fit(data)
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(14,7))

ax1.set_title('K Means')

ax1.scatter(x1,x2,c=model.labels_,cmap='rainbow')

ax2.set_title("Original")

ax2.scatter(x1,x2,c=y,cmap='rainbow')
model.labels_