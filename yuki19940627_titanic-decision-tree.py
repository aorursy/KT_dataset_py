# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline 

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_validate

from sklearn.model_selection import GridSearchCV

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

submission = pd.read_csv('../input/gender_submission.csv')
train.head()
train.info()
train.describe()
train = train.drop(columns=['Name', 'Ticket','Cabin'])

test = test.drop(columns=['Name', 'Ticket', 'Cabin'])
pd.crosstab(train['Sex'], train['Survived'], margins=True)
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()
train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()
pd.crosstab(train['Embarked'], train['Survived'],margins=True)
train = pd.get_dummies(train)

test = pd.get_dummies(test)
print(train.isnull().sum())

print(test.isnull().sum())
train['Age'] = train['Age'].fillna(train['Age'].median())

train.isnull().sum()
test['Age'] = test['Age'].fillna(test['Age'].median())

test['Fare'] = test['Fare'].fillna(test['Fare'].median())

train.isnull().sum()
trainX = train.drop(columns=['Survived'])

y = train['Survived']
testX = test.copy()

testX.head()
clf = DecisionTreeClassifier()
params = {'max_depth': list(range(2,11)), 'min_samples_leaf': [5,10,20,50,100,500]}
GS = GridSearchCV(clf, params, cv=5, scoring="roc_auc",n_jobs=-1)

GS.fit(trainX,y)
GS.best_params_
GS.score(trainX, y)
params = {'max_depth': list(range(5,15)), 'min_samples_leaf': [16,17,18,19,20,21,22,23,24]}

GS = GridSearchCV(clf, params, cv=5, scoring="roc_auc",n_jobs=-1)

GS.fit(trainX,y)
GS.best_params_
prediction = GS.predict_proba(testX)

prediction = prediction[:,1]

prediction = [round(i) for i in prediction]
mySubmission = pd.DataFrame({ 'PassengerId': testX['PassengerId'],

                            'Survived': prediction }, dtype=int)

mySubmission.to_csv("submission.csv",index=False)