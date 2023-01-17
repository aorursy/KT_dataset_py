# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Reading the csv into a dataframe
df = pd.read_csv("../input/Iris.csv")
df.head(10)
# Describing the dataset
df.describe(include='all')
# Only get iris features (no ID/Species)
df_features = df.drop(['Id', 'Species'], axis=1)

plt.figure(figsize=(10,6))
sns.heatmap(df_features.corr(), cmap='viridis', annot=True)
# Show violin plots of all the features
plt.figure(figsize=(24,6))

for i, col in enumerate(df_features.columns):
    #print(i)
    #print(col)
    ax = plt.subplot(1, 4, i+1)
    sns.violinplot(x='Species', y=col, data=df)
    plt.title("Iris Species by " + col)
# How can we tell versicolor and virginica apart?
sns.pairplot(df.drop('Id', axis=1), hue='Species')
# Set the Species to numbers - 0, 1, 2
species_map = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}
df['target'] = df.Species.map(species_map)

X = df.drop(['Id','Species','target'], axis=1)
y = df.target

X.head()
#y.head()
# Split X/y into train/test datasets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0)
# Make sure shapes are what we'd expect
print("Train shapes:")
print(X_train.shape)
print(y_train.shape)

print("\nTest shapes:")
print(X_test.shape)
print(y_test.shape)
#Run kNN, fit to training data
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)
# Using our kNN model...predict "target" for X_test
y_pred = knn.predict(X_test)

# What were our predictions?
y_pred

#97.4% right!!!
knn.score(X_test, y_test)