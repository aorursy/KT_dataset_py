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

        

import matplotlib.pyplot as plt

import seaborn as sn

sn.set()

# Any results you write to the current directory are saved as output.
dados = pd.read_csv("/kaggle/input/wholesale-customers-data-set/Wholesale customers data.csv")

dados.head()
dados.columns
X= dados.iloc[:,2:]

X.head()
X.describe()
X_array= X.values
X_array
from sklearn.cluster import KMeans

wcss=[]

for i in range(1,11):

    Kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)

    Kmeans.fit(X_array)

    wcss.append(Kmeans.inertia_)

plt.plot(range(1,11),wcss)

plt.title('Metodo Cotuvelo - Elbow Method')

plt.xlabel('numero de clustere')

plt.ylabel('WCSS')

plt.show()

    
Kmeans=KMeans(n_clusters=4,init='k-means++',max_iter=300,n_init=10,random_state=0)

dados['cluster']=Kmeans.fit_predict(X_array)

dados.head()

dados.groupby("cluster").agg('mean').plot.bar(figsize=(10,7.5))

plt.title("gastos por cluster")