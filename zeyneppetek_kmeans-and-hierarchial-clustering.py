# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
data.head()

y=data["age"].values
x1=data["trestbps"].values
x2=data["chol"].values

x=np.concatenate((x1,x2),axis=0)
#visualization
plt.scatter(x1,y)
plt.scatter(x2,y)
plt.show()

#KMeans Clustering
from sklearn.cluster import KMeans
wcss=[] #wcss, provide finding optimum 'k' clustering number.
for k in range(1,15):
    Kmeans=KMeans(n_clusters=k)
    Kmeans.fit(data)
    wcss.append(Kmeans.inertia_) #inertia: For each k value, shows wscc values. 
plt.plot(range(1,15),wcss)
plt.xlabel("number of k values")
plt.ylabel("wcss")
plt.show()

#model for k=2

kmeans2=KMeans(n_clusters=2)
clusters=kmeans2.fit_predict(data)

data["label"]=clusters #we define labels for clusters and we will be completed classification.
plt.scatter(data["chol"],data["trestbps"],c = clusters)
plt.xlabel("trestbps")
plt.ylabel("chol")
plt.show()    


from scipy.cluster.hierarchy import linkage,dendrogram
merg=linkage(data,method="ward") #ward:it provides minimum variant clustering.
dendrogram(merg,leaf_rotation=90)
plt.xlabel("data point")
plt.ylabel("Euclidean Distance")
plt.show()
from sklearn.cluster import AgglomerativeClustering 
#AgglomerativeClustering, it is process that points that close each make groups and reach one gruop.
cluster=hierarchial_cluster.fit_predict(data)
data["label"]=cluster

plt.scatter(data["chol"],data["trestbps"],c = clusters)
plt.xlabel("trestbps")
plt.ylabel("chol")
plt.show()    
