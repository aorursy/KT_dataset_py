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
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
%matplotlib inline
x = -2 * np.random.rand(150, 2)
x1 = 1+3 * np.random.rand(50, 2)
x2 = 10+2 * np.random.rand(50, 2)
x[50:100, :] = x1
x[100:150, :] = x2
plt.scatter(x[:, 0], x[:, 1], s=50, c='b')
plt.show()
Kmean = KMeans(n_clusters=3)
Kmean.fit(x)
print(Kmean)
cluster_centers = Kmean.cluster_centers_
print(cluster_centers)
plt.scatter(x[:, 0], x[:, 1], s=50, c='b')
plt.scatter(cluster_centers[0, 0], cluster_centers[0, 1], s=200, c='red')
plt.scatter(cluster_centers[1, 0], cluster_centers[1, 1], s=200, c='green')
plt.scatter(cluster_centers[2, 0], cluster_centers[2, 1], s=200, c='yellow')
plt.show()
print(Kmean.labels_)
sample_test = np.array([-12, -10])
sample_test = sample_test.reshape(1, -1)
type = Kmean.predict(sample_test)
print(type)