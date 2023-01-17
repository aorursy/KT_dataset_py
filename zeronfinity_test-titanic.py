# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

y = train.Survived

predictors = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']

X = train[predictors]
X_t = test[predictors]

enX = pd.get_dummies(X)
enX_t = pd.get_dummies(X_t)
X, X_t = enX.align(enX_t, join='left', axis=1)

print(X.columns)

# X = X.fillna(X.mean())
# X_t = X_t.fillna(X_t.mean())

missingcol = [col for col in X.columns 
                                if X[col].isnull().any()]
X = X.drop(missingcol, axis=1)
X_t = X_t.drop(missingcol, axis=1)
missingcol = [col for col in X_t.columns 
                                if X_t[col].isnull().any()]
X = X.drop(missingcol, axis=1)
X_t = X_t.drop(missingcol, axis=1)
print(X.columns)

train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)

clf = LinearSVC(random_state=0)
clf.fit(train_X, train_y)
pred_y = clf.predict(val_X)
print(mae(val_y, pred_y))

y_t = clf.predict(X_t)
#print(y_t)
sub = pd.DataFrame({'PassengerId': X_t.PassengerId, 'Survived': y_t})
sub.to_csv('submission.csv', index=False)