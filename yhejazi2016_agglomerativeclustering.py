

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import pandas as pd

#breast_cancer_dataset = pd.read_csv("../input/breast_cancer_dataset.csv")

data = pd.read_csv("../input/new_dataset01.csv")

data[:10]
#X Data

X = data.drop(['group_id'], axis=1, inplace=False)

print('X Data is \n' , X.head())

#print('X shape is ' , X.shape)
y = data['group_id']

y[:10]
from sklearn.model_selection import train_test_split

#Splitting data



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=50, shuffle =False)



#Splitted Data

#print('X_train shape is ' , X_train.shape)

#print('X_test shape is ' , X_test.shape)

#print('y_train shape is ' , y_train.shape)

#print('y_test shape is ' , y_test.shape)
from sklearn.cluster import AgglomerativeClustering

import scipy.cluster.hierarchy as sch

import matplotlib.pyplot as plt



#Applying AggClusteringModel Model 



'''

sklearn.cluster.AgglomerativeClustering(n_clusters=2, affinity='euclidean’, memory=None, connectivity=None, 

                                        compute_full_tree='auto’, linkage=’ward’,pooling_func=’deprecated’)

'''



AggClusteringModel = AgglomerativeClustering(n_clusters=5,affinity='euclidean',# it can be euclidean وl1,l2,manhattan,cosine,precomputed

                                             linkage='ward')# it can be complete,average,single



y_pred_train = AggClusteringModel.fit_predict(X_train)

y_pred_test = AggClusteringModel.fit_predict(X_test)

#y_pred_train



#draw the Hierarchical graph for Training set

dendrogram = sch.dendrogram(sch.linkage(X_train, method = 'ward'))# it can be complete,average,single

plt.title('Training Set')

plt.xlabel('X Values')

plt.ylabel('Distances')

plt.show()
#draw the Hierarchical graph for Test set

dendrogram = sch.dendrogram(sch.linkage(X_test, method = 'ward'))# it can be complete,average,single

plt.title('Test Set')

plt.xlabel('X Value')

plt.ylabel('Distances')

plt.show()
from sklearn import metrics

def purity_score(y_true, y_pred):

    # compute contingency matrix (also called confusion matrix)

    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)

    # return purity

    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 

print(purity_score(y_test, y_pred_test))
print("Adjusted Rand Index: " , metrics.adjusted_rand_score(y_test, y_pred_test))

print("Adjusted Mutual Information:" , metrics.adjusted_mutual_info_score(y_test, y_pred_test))
