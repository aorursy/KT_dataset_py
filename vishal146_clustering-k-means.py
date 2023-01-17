# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/uci-wholesale-customers-data/Wholesale customers data.csv")
data.head()
data.describe()
#Scaling Data

from sklearn.preprocessing import StandardScaler

new_data= StandardScaler().fit_transform(data)

pd.DataFrame(new_data).head()

#Scaled data Description

pd.DataFrame(new_data).describe()
#K means++ intialization using k-means function from sklearn.cluster and then fitting in it.

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, init="k-means++")

kmeans.fit(new_data)
kmeans.inertia_ #inerrtia of the kmeans
#multiple k means algorithms and storing values of inertia in a list

list = []

for cluster in range(1,20):

    kmeans = KMeans(n_jobs=-1, n_clusters=cluster, init='k-means++')

    kmeans.fit(new_data)

    list.append(kmeans.inertia_)

    

    print(list)

    
#converting the list into dataframe and then plotting it list contains Sum of squared distance betwen centroids called inertia

df=pd.DataFrame({'Cluster': range(1,20), 'inertia' : list})

plt.figure(figsize=(14,8))

plt.plot(df['Cluster'],df['inertia'])

plt.xlabel('Clusters')

plt.ylabel('inertia')

plt.show()
# k means using 5 clusters and k-means++ initialization

kmeans = KMeans(n_jobs = -1, n_clusters = 5, init='k-means++')

kmeans.fit(new_data)

pred = kmeans.predict(new_data)
frame = pd.DataFrame(new_data)

frame['cluster'] = pred

frame['cluster'].value_counts()