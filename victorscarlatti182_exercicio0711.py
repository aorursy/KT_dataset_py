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



import sklearn.datasets as dt

import matplotlib.pyplot as plt

import seaborn as sn

sn.set()
dt = pd.read_csv("/kaggle/input/carsdata/cars.csv",na_values=' ')
dt.head()
plt.scatter(dt.loc[:50,'mpg'],

           dt.loc[:50,' brand'],

           color='red',

           marker='+')

plt.scatter(dt.loc[51:100,'mpg'],

           dt.loc[51:100,' brand'],

           color='blue',

           marker='o')

plt.scatter(dt.loc[101:150,'mpg'],

           dt.loc[101:150,' brand'],

           color='green',

           marker='*')

plt.ylabel("mpg")

plt.xlabel("brand")

plt.title("Classes")

plt.show()
X = dt['mpg']

from sklearn.cluster import KMeans

km = KMeans(n_clusters=3, random_state=0)

km.fit(X)

pred = km.predict(X)

res = km.predict(X)

res = pd.DataFrame(np.array(dt[' brand']), columns=["brand"])

res['predict']=pred

res.sample(15)

acertos = res[res['predict']==res["brand"]].count()

acertos
dt.columns
dt.columns=['mpg','cylinders','cubicinches','hp','weightlbs','time-to-60','year','brand']
dt = dt.dropna()

dt['cubicinches'] =  dt['cubicinches'].astype(int)

dt['weightlbs'] =  dt['weightlbs'].astype(int)
x = dt.iloc[:,:7]

x.head()
x_array = x.values

x_array
from sklearn.cluster import KMeans

wcss = []

for i in range(1,11):

    kmeans =KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)

    kmeans.fit(x_array)

    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)

plt.title('Método Cotuvelo - Elbow Method')

plt.xlabel('Número de Clusters')

plt.ylabel('WCSS')

plt.show()
kmeans =KMeans(n_clusters=4,init='k-means++',max_iter=300,n_init=10,random_state=0)

dt['clusters'] = kmeans.fit_predict(x_array)

dt.head()

dt.groupby("clusters").agg('mean').plot.bar(figsize=(10,7.5))

plt.title("Gastos por Cluster")