## Importing Libraries

import pandas as pd

import numpy as np  

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LogisticRegression

from sklearn import metrics
## Loading dataset

data = pd.read_csv("../input/logistic-regression/Iris.csv")

data.head()
## Checking Size of the data

data.shape
## Checking the datatype 

data.info()
## Checking if there any null values exists

data.isnull().sum()
## Descriptive statistics of the data

data.describe()
## Count of each group of species

data['Species'].value_counts()
temp = data.drop('Id', axis=1)

visual = sns.pairplot(temp, hue='Species', markers='+')

plt.show()
X = data.drop(['Id', 'Species'], axis=1)

y = data['Species']





print(X.head())

print(X.shape)





print(y.head())

print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
logreg = LogisticRegression()

logreg.fit(X_train, y_train)    # fit the model

y_pred = logreg.predict(X_test)   # predict the model

y_pred
## Accuracy score

print(metrics.accuracy_score(y_test, y_pred))
## Predicting with manual values

logreg.predict([[5,1,3,2]])[0]