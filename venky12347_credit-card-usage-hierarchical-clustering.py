# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas                  as     pd

import numpy                   as     np

from   sklearn.cluster         import AgglomerativeClustering, KMeans

import sklearn.datasets

from   scipy.cluster.hierarchy import dendrogram, linkage

from   matplotlib              import pyplot as plt

%matplotlib inline

np.set_printoptions(precision=5, suppress=True)  # suppress scientific float notation
card = pd.read_csv("../input/CreditCardUsage.csv")
card.describe().T
mean_value=card['CREDIT_LIMIT'].mean()

card['CREDIT_LIMIT']=card['CREDIT_LIMIT'].fillna(mean_value)
mean_value=card['MINIMUM_PAYMENTS'].mean()

card['MINIMUM_PAYMENTS']=card['MINIMUM_PAYMENTS'].fillna(mean_value)
df = card.drop('CUST_ID', axis=1)
df.head(3)
import scipy.cluster.hierarchy as sch

plt.figure(figsize=(15,6))

dendrogram = sch.dendrogram(sch.linkage(df.iloc[:,[0,2]].values, method = 'ward'))

plt.title('Dendrogram', fontsize = 20)

plt.xlabel('BALANCE AND PURCHASES')

plt.ylabel('Euclidean  Distance')

plt.show()
import scipy.cluster.hierarchy as sch

plt.figure(figsize=(15,6))

dendrogram = sch.dendrogram(sch.linkage(df.iloc[:,[0,13]].values, method = 'ward'))

plt.title('Dendrogram', fontsize = 20)

plt.xlabel('BALANCE AND PYMT')

plt.ylabel('Euclidean  Distance')

plt.show()
import scipy.cluster.hierarchy as sch

plt.figure(figsize=(15,6))

dendrogram = sch.dendrogram(sch.linkage(df.iloc[:,[0,4]].values, method = 'ward'))

plt.title('Dendrogram', fontsize = 20)

plt.xlabel('BALANCE AND INSTALLMENT')

plt.ylabel('Euclidean  Distance')

plt.show()
import scipy.cluster.hierarchy as sch

plt.figure(figsize=(15,6))

dendrogram = sch.dendrogram(sch.linkage(df.iloc[:,[2,5]].values, method = 'ward'))

plt.title('Dendrogram', fontsize = 20)

plt.xlabel('BALANCE AND CASH ADVANCE')

plt.ylabel('Euclidean  Distance')

plt.show()
import scipy.cluster.hierarchy as sch

plt.figure(figsize=(15,6))

dendrogram = sch.dendrogram(sch.linkage(df.iloc[:,[2,13]].values, method = 'ward'))

plt.title('Dendrogram', fontsize = 20)

plt.xlabel('PURCHASE AND PAYMENT')

plt.ylabel('Euclidean  Distance')

plt.show()
X_BAL_PUR = df.iloc[:,[0,2]].values

X_BAL_INSTALL_PUR=df.iloc[:,[0,4]].values

X_PUR_CASH_ADVANCE=df.iloc[:,[2,5]].values

X_PUR_PAYMENT=df.iloc[:,[2,13]].values

X_BAL_PYMT=df.iloc[:,[0,13]].values
from sklearn.cluster import KMeans



wcss = []

for i in range(1, 11):

    km = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

    km.fit(df)

    wcss.append(km.inertia_)

    

plt.plot(range(1, 11), wcss)

plt.title('The Elbow Method', fontsize = 20)

plt.xlabel('No. of Clusters')

plt.ylabel('wcss')

plt.show()
from sklearn.cluster import AgglomerativeClustering



hc_7 = AgglomerativeClustering(n_clusters = 7, affinity = 'euclidean', linkage = 'ward')

hc_5 = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')





y_hc = hc.fit_predict(df)

y_hc_bal_pur = hc_5.fit_predict(X_BAL_PUR)

y_hc_bal_pymt = hc_5.fit_predict(X_BAL_PYMT)

y_hc_pur_pymt = hc_5.fit_predict(X_PUR_PAYMENT)



plt.scatter(X_BAL_PUR[y_hc_bal_pur == 0, 0], X_BAL_PUR[y_hc_bal_pur == 0, 1], s = 100, c = 'yellow', label = 'Low Balance,Low Purchase')

plt.scatter(X_BAL_PUR[y_hc_bal_pur == 1, 0], X_BAL_PUR[y_hc_bal_pur == 1, 1], s = 100, c = 'pink', label = 'High Balance,High Purchase')

plt.scatter(X_BAL_PUR[y_hc_bal_pur == 2, 0], X_BAL_PUR[y_hc_bal_pur == 2, 1], s = 100, c = 'cyan', label = 'Medium Balance,Medium Purchase')

plt.scatter(X_BAL_PUR[y_hc_bal_pur == 3, 0], X_BAL_PUR[y_hc_bal_pur == 3, 1], s = 100, c = 'magenta', label = 'High Balance,Low Purchase')

plt.scatter(X_BAL_PUR[y_hc_bal_pur == 4, 0], X_BAL_PUR[y_hc_bal_pur == 4, 1], s = 100, c = 'orange', label = 'Low Balance,Medium Purchase')

plt.scatter(X_BAL_PUR[y_hc_bal_pur == 5, 0], X_BAL_PUR[y_hc_bal_pur == 5, 1], s = 100, c = 'red', label = 'Medium Balance,Medium Purchase')

plt.scatter(X_BAL_PUR[y_hc_bal_pur == 6, 0], X_BAL_PUR[y_hc_bal_pur == 6, 1], s = 100, c = 'violet', label = 'Low Balance,Low Purchase')

plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 50, c = 'blue' , label = 'centroid')



plt.axhspan(ymin=20,ymax=25,xmin=0.4,xmax=0.96,alpha=0.3,color='yellow')

plt.title('Hierarchial Clustering - Clustering of Balance Vs Purchase', fontsize = 20)

plt.xlabel('Balance')

plt.ylabel('Purchase')

plt.legend()

plt.grid()

plt.show()