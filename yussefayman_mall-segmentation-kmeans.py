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
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
train_data = pd.read_csv("/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv")
print(train_data.head())
#train_data.info()
#train_data.shape
#no missing Values
#train_data.isnull().sum()
features = ["Spending Score","Annual Income"]
X =train_data.iloc[:, [3,4]].values
#print(X)
#Determine The optimal K 
k_means_values = []
#sklearn use 8 clusters as default
for i in range(1,9):
    kmean = KMeans(i,init='k-means++',random_state=0)
    kmean.fit(X)
    k_means_values.append(kmean.inertia_)

plt.plot(range(1,9), k_means_values)
plt.title('The Elbow Curve')
plt.xlabel('num of clusters')
plt.ylabel('Data points')
plt.show()
model = KMeans(n_clusters= 5, init='k-means++', random_state=0)
prediction = model.fit_predict(X)
labels = model.labels_
centers = model.cluster_centers_
list_of_colors = ["green","yellow","blue","magenta","red"]

for i in range(len(X)):
    plt.scatter(X[i][0],X[i][1],s=70,c=list_of_colors[labels[i]])
plt.scatter(centers[:,0],centers[:,1],marker="X",s=250,linewidths=5, c = 'black')
plt.show()
