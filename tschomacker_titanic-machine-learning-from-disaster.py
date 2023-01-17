import numpy as np

import pandas as pd

from keras.models import Sequential

from keras.layers import Dense, Activation

from sklearn import tree

import graphviz



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
gender_submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

gender_submission
#read from csv file

X_train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

#sequence of labels (0,1) if the passenger at index i in train had survived

Y_train = X_train['Survived']

#remove the labels

X_train.pop('Survived')

#remove name and ticket (passenger get referenced by PassengerID)

X_train.pop('Name')

X_train.pop('Ticket')

X_test = test

X_test.pop('Name')

X_test.pop('Ticket')

# remove columns with NaN values

X_train = X_train.dropna(axis=1)

X_test = X_test.dropna(axis=1)

X_test.pop('Embarked')

X_train.pop('Fare')

# one-hot encoding for all categorical data

X_train =  pd.get_dummies(X_train)

X_test =  pd.get_dummies(X_test)

X_train_backup = X_train

Y_train_backup = Y_train

X_test_backup = X_test

print(X_train.keys())

print(X_test.keys())
print(X_train.head())
# create a numpy array

X_train_array = X_train.to_numpy()

X_train.plot.scatter(x='Pclass',

                      y='SibSp',)
from sklearn.tree import DecisionTreeClassifier

X_train = X_train_backup

Y_train = Y_train_backup

X_test = X_test_backup

clf = DecisionTreeClassifier().fit(X_train, Y_train)

print('Accuracy of Decision Tree classifier on training set: {:.2f}'

     .format(clf.score(X_train, Y_train)))

print('Accuracy of Decision Tree classifier on test set: {:.2f}'

     .format(clf.score(X_test, Y_test)))

X_test['Survived'] = Y_test

print(test)

test.to_csv('submission.csv', index=False)
from sklearn.neighbors import KNeighborsClassifier

X_train = X_train_backup

Y_train = Y_train_backup

X_test = X_test_backup

knn = KNeighborsClassifier()

knn.fit(X_train, Y_train)

print('Accuracy of K-NN classifier on training set: {:.2f}'

     .format(knn.score(X_train, Y_train)))

print('Accuracy of K-NN classifier on test set: {:.2f}'

     .format(knn.score(X_test, Y_test)))