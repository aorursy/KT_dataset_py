# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

#breast_cancer_dataset = pd.read_csv("../input/agglomerativeclustering-ds/breast_cancer_dataset.csv")

data = pd.read_csv("../input/agglomerativeclustering-ds/new_dataset01.csv")




X= data.drop('group_id',axis=1)

y= data.group_id

X.shape

#  init = k-means++ or random

kmeans = KMeans(n_clusters =5)       

y_kmeans = kmeans.fit_predict(X)

y_kmeans



#way to show best cluster number 

from sklearn import metrics

score = []

for n in range(2,5):

    kmeans = KMeans(n_clusters= n )

    kmeans.fit(X)

    result = kmeans.labels_

    print(n , '    '  , silhouette_score(X , result))

    print("Adjusted Rand Index: " , metrics.adjusted_rand_score(y, y_kmeans))

    print("Adjusted Mutual Information: " , metrics.adjusted_mutual_info_score(y, y_kmeans))

    print("silhouette_score: " ,silhouette_score(X , result))

    score.append(silhouette_score(X , result))
    

plt.plot(range(2,5) , score)

plt.show()

def purity_score(y_true, y_pred):

    # compute contingency matrix (also called confusion matrix)

    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)

    # return purity

    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 

print(purity_score(y, y_kmeans))