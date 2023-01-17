#load libraries and packages

from sklearn.metrics import make_scorer, accuracy_score ,classification_report,f1_score

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler

from yellowbrick.cluster import KElbowVisualizer

from sklearn.cluster import AgglomerativeClustering

import scipy.cluster.hierarchy as sch

from sklearn.cluster import KMeans

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

import sklearn.metrics as sk

from sklearn import preprocessing

import matplotlib.pylab as pylab

import matplotlib.pyplot as plt

from pandas import get_dummies

import matplotlib as mpl

import seaborn as sns

import pandas as pd

import numpy as np

import matplotlib

import warnings

import sklearn

import scipy

import numpy

import json

import sys

import csv

import os



from IPython.core.interactiveshell import InteractiveShell         #to display multiple outputs in same cell

InteractiveShell.ast_node_interactivity = "all"

data = pd.read_csv("../input/wine-quality/winequality.csv")

data
#identify datatypes of features

data.dtypes
#statistical analysis

data.describe()
#find number of wines that are classified 'good' and 'bad'

data.good.value_counts()
#plotting histogram of all features

data.hist(figsize=(15,20))
#regression plot of chlorides vs quality

f,ax=plt.subplots(figsize=(10,10))

sns.regplot(x='chlorides',y='quality',data=data)

plt.title('regression plot of chlorides and quality')
#regression plot of alcohol vs quality

f,ax=plt.subplots(figsize=(10,10))

sns.regplot(x='alcohol',y='quality',data=data)

plt.title('regression plot of alcohol and quality')
#countplot of quality based on goodness feature

sns.catplot(x='quality',data=data,height=5,aspect=3,hue='good',kind='count')

plt.title('countplot of qulaity')
#correltaion matrix/heatmap

data.corr()

f,ax=plt.subplots(figsize=(10,10))

sns.heatmap(data.corr())

plt.title('heat map')
#remove unwanted columns

df=data.drop(['quality','free sulfur dioxide'],axis=1)
data.isnull().sum()
#one hot encoding

df=pd.get_dummies(data)
#creating x and y dataframes

x=df.drop('good',axis=1)

y=df[['good']]
#split into train and test data

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=100)
# Perform random forest with grid search to optimize model

rfc=RandomForestClassifier(random_state=42)

param_grid = { 

    'n_estimators': [200,300,400],

    'max_features': ['auto', 'sqrt', 'log2'],

    'max_depth' : [4,5,6,7,8],

    'criterion' :['gini', 'entropy']

}

random = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5).fit(x_train, y_train)

random.predict(x_train)

random.predict(x_test)
print('train model metrics')

confusion_matrix(y_train,random.predict(x_train))

print('accuracy-',round(random.score(x_train,y_train) * 100, 2))

print('f1 score-',f1_score(y_train,random.predict(x_train)))

print(" ")

print('test model metrics')

confusion_matrix(y_test,random.predict(x_test))

print('accuracy-',round(random.score(x_test,y_test) * 100, 2))

print('f1 score-',f1_score(y_test,random.predict(x_test)))

#predict probabilites

rf_prob=random.predict_proba(x_train)

rf_prob=rf_prob[:,1]

false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_train, rf_prob)

print('auc_score for random forest(train): ', roc_auc_score(y_train, rf_prob))



# Plot ROC curves

plt.subplots(1, figsize=(5,5))

plt.title('Receiver Operating Characteristic(train) - random forest')

plt.plot(false_positive_rate1, true_positive_rate1)

plt.plot([0, 1], ls="--")

plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()





rf_prob_test=random.predict_proba(x_test)

rf_prob_test=rf_prob_test[:,1]

false_positive_rate2, true_positive_rate2, threshold2 = roc_curve(y_test, rf_prob_test)

print('auc_score for random forest(test): ', roc_auc_score(y_test, rf_prob_test))



# Plot ROC curves

plt.subplots(1, figsize=(5,5))

plt.title('Receiver Operating Characteristic(test) - random forest')

plt.plot(false_positive_rate2, true_positive_rate2)

plt.plot([0, 1], ls="--")

plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()



#correltaion matrix/heatmap

data.corr()

f,ax=plt.subplots(figsize=(10,10))

sns.heatmap(df.corr())

plt.title('heat map')
#one hot encoding to deal with categorical data

x_onehot=pd.get_dummies(x)
#data normalisation to bring data of every feature on a same scale

x_scale = StandardScaler().fit_transform(x_onehot)
#elbow method with inertia to find n clusters

wcss = []

for i in range(1, 11):

    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)

    kmeans.fit(x_scale)

    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)

plt.title('Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
#check silhouette score

# Instantiate a scikit-learn K-Means model

model = KMeans(random_state=0)



# Instantiate the KElbowVisualizer with the number of clusters and the metric 

visualizer = KElbowVisualizer(model, k=(2,10), metric='silhouette', timings=False)



# Fit the data and visualize

visualizer.fit(x_scale)    

visualizer.poof()   
#applying kmeans algorith

kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=0)

pred_y = kmeans.fit_predict(x_scale)

plt.scatter(x_scale[:,0],x_scale[:,1],c=pred_y,cmap='viridis')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')

plt.show()

#calculating davies bouldin score

sklearn.metrics.davies_bouldin_score(x_scale,pred_y)
#comparisons using mean

x_kmeans=x.copy()

x_kmeans['labels']=pred_y

x_kmeans.groupby('labels').mean()

data.groupby('color').mean()
#hierarchical clustering-plotting dendrogram

dendrogram = sch.dendrogram(sch.linkage(x_scale, method='ward'))
#applying agglomerative clustering algorithm

model = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')

model.fit_predict(x_scale)

labels = model.labels_

#plotting clusters on scatter plot

plt.figure(figsize=(10, 7))

plt.scatter(x_scale[labels==0, 0], x_scale[labels==0, 1], s=50, marker='o', color='red')

plt.scatter(x_scale[labels==1, 0], x_scale[labels==1, 1], s=50, marker='o', color='blue')

sklearn.metrics.davies_bouldin_score(x_scale,labels)
x_hrcl=x.copy()

x_hrcl['labels']=labels

x_hrcl.groupby('labels').mean()

data.groupby('color').mean()
