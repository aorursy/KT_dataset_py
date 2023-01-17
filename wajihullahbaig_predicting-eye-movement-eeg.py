# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import itertools

import matplotlib.pyplot as plt





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

inputData = pd.read_csv('../input/eye movement.csv', delimiter=',')

print(inputData.dtypes)

print(inputData.columns)

print("Data shape:",inputData.shape)

print(inputData.head())

print(inputData.describe())

print(inputData.info())
print(inputData.isnull().sum())
plt.figure( figsize=(10,10))

inputData['eyeDetection{0,1}'].value_counts().plot.pie(autopct="%1.1f%%")

plt.title("Data division on eye movement",fontsize=10)

plt.show()
fig = plt.figure(figsize=(20, 15))

ax = fig.gca()

inputData.plot(ax=ax,kind='density',subplots=True,sharex=False)

plt.show()
fig = plt.figure(figsize = (10,10))

ax = fig.gca()

sns.heatmap(inputData.corr(), annot=True, fmt=".2f")

plt.title("Correlation",fontsize=5)

plt.show()
eyes0 = inputData.loc[inputData["eyeDetection{0,1}"]==0].sample(frac=0.01)

eyes1 = inputData.loc[inputData["eyeDetection{0,1}"]==1].sample(frac = 0.01)

v = pd.concat([eyes0,eyes1])





sns.pairplot(data=v,hue="eyeDetection{0,1}")

plt.title("Skewness",fontsize =10)

plt.show()
from sklearn.model_selection import train_test_split

X = inputData.iloc[:,0:2].copy()

y = inputData["eyeDetection{0,1}"].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
# Grid search to find best paramaters

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold



param_grid = [

  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},

  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}, # Ran too slowly with these

  #{'kernel':['linear']},

  #{'kernel':['rbf'], 'gamma':[0.001, 0.0001]}

]



estimator = SVC()

clf = GridSearchCV(estimator=estimator, param_grid=param_grid, n_jobs=-1)

cross_val_score(clf, X_train, y_train)
# Testing

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
#Import scikit-learn metrics module for accuracy calculation

from sklearn import metrics



# Model Accuracy: how often is the classifier correct?

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))