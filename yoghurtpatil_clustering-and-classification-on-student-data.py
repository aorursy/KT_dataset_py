import os

import math

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from mpl_toolkits.mplot3d import Axes3D

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
# Load the data from the excel file and look at column names

os.chdir("/kaggle/input")

orig = pd.read_csv('user-knowledge/User Knowledge.csv')

orig.columns
# Keep only the columns containing the data about student's knowledge

knowledge = orig.iloc[:,:5]

knowledge.head()
# Plot histograms of the featuers to visualize the data

knowledge.hist(bins=50, figsize = (8,8))

plt.show()
# Perform k-Means Clustering with values of k from 1 to 10 and plot k v/s Within Cluster Sum of Squares

wcss = []

for i in range(1, 11):

    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=400, n_init=20, random_state=0)

    kmeans.fit(knowledge)

    wcss.append(kmeans.inertia_)



plt.plot(range(1, 11), wcss)

plt.title('Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
# K-Means Clustering with 3 clusters

kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=400, n_init=20, random_state=0)

kmeans.fit(knowledge)

k_class = kmeans.predict(knowledge)
# Using PCA and filtering 3 principal components for data visualization

pca = PCA(n_components=3)

principalComponents = pca.fit_transform(knowledge)

PDF = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2', 'PC3'])
# Add a column 'Class' to the data sets

PDF.loc[:, 'Cluster'] = pd.Series(k_class)

knowledge_class = knowledge.copy()

knowledge_class['Class'] = k_class
# Count of points in each cluster

PDF['Cluster'].value_counts()
# Assign a color to each cluster

PDF['Color'] = PDF['Cluster'].map({0 : 'red', 1 : 'blue', 2 : 'green'})
# Plot the first 2 principal components and color by cluster

a1 = PDF['PC1']

a2 = PDF['PC2']

a3 = PDF['PC3']

c1 = PDF['Color']

plt.scatter(a1, a2, c = c1, alpha=0.3, cmap='viridis')
# 3-D plot of the data using 3 principal components

fig = plt.figure()

ax = Axes3D(fig)

ax.scatter(a1, a2, a3, alpha = 0.4, c = c1)
knowledge_class.groupby(['Class']).mean()
# Slipt the data into train and test data sets

X = knowledge_class.iloc[:, :-1]

Y = knowledge_class.iloc[:, -1]

xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size = 0.25, random_state = 0)
# KNN for various values of k and plot of k v/s accuracy

from sklearn.neighbors import KNeighborsClassifier

accuracy = []

for i in range(1,12):

    knn = KNeighborsClassifier(n_neighbors = i).fit(xTrain, yTrain)

    accuracy.append(knn.score(xTest, yTest))



plt.plot(range(1,12), accuracy)

plt.xlabel('k')

plt.ylabel('Accuracy') 

plt.title('k v/s Accuracy for KNN')
# KNN model and evaluation for optimal value of k (8 in this case)

knn = KNeighborsClassifier(n_neighbors = accuracy.index(max(accuracy))+1).fit(xTrain, yTrain)

knn_predictions = knn.predict(xTest)

knn_accuracy = knn.score(xTest, yTest)

knn_accuracy
knn_CM = confusion_matrix(yTest, knn_predictions) # KNN Confusion Matrix

knn_CM
# Decision Tree Classifier and evaluation for optimal value of k

from sklearn.tree import DecisionTreeClassifier

dtree_model = DecisionTreeClassifier(max_depth = 2).fit(xTrain, yTrain) 

dtree_predictions = dtree_model.predict(xTest)

dt_accuracy = dtree_model.score(xTest, yTest)

dt_accuracy
DT_CM = confusion_matrix(yTest, dtree_predictions) # Decision Tree confusion Matrix

DT_CM
# Gaussian Naive Bayes model and evaluation for optimal value of k

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB().fit(xTrain, yTrain)

gnb_predictions = gnb.predict(xTest)

gnb_accuracy = gnb.score(xTest, yTest)

gnb_accuracy
NB_CM = confusion_matrix(yTest, gnb_predictions) # Naive Bayes confusion Matrix

NB_CM