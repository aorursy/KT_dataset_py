# Credit Card Cluster Problem with K-Means

# Importing Preprocessing Libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# Importing Datasets
dataset = pd.read_csv('../input/CC GENERAL.csv')
X = dataset.iloc[:, 1:].values
# Dataset Contains Multiple Missing values
# Replacing Missing Value by Most Repeated/Frequent Number in that column
# Use Imputer with strategy 'most_frequent'
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="most_frequent", axis=0)
imputer = imputer.fit(X)
X = imputer.transform(X)

# Applying Feature Scalling with StandardScaler
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# For Finding Optimal Number of Cluster use Elbow Method 
from sklearn.cluster import KMeans
wcss = []
for i in range(1,18):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0).fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,18), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Cluster')
plt.ylabel('WCSS')
plt.show()

# Apply K-Means Again With Optimal Number of Cluster that we got from Elbow method i.e. 8
kmeans = KMeans(n_clusters=8, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X)
# Finally Append new Column i.e Cluster to Actual Dataset
dataset['Cluster'] = y_kmeans
dataset.head()