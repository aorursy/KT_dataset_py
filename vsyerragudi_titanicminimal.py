# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.tree import DecisionTreeClassifier # For intelligence

#Import datasets
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
#Clean
train.Sex.replace({'male':1,'female':2}, inplace=True)
train.Embarked.replace({'C':1,'Q':2,'S':3}, inplace=True)
train.drop(columns=['Cabin'],inplace=True)
train.rename(columns={'Pclass':'PassengerClass','SibSp':'SiblingOrSpouses','Parch':'ParentsOrChildren'}, inplace=True)
test.rename(columns={'Pclass':'PassengerClass','SibSp':'SiblingOrSpouses','Parch':'ParentsOrChildren'}, inplace=True)
train=train.dropna()

test.Sex.replace({'male':1,'female':2}, inplace=True)
test.Embarked.replace({'C':1,'Q':2,'S':3}, inplace=True)
test.drop(columns=['Cabin'],inplace=True)
test.rename(columns={'Pclass':'PassengerClass','SibSp':'SiblingOrSpouses','Parch':'ParentsOrChildren'}, inplace=True)
test.set_index('PassengerId',inplace=True)

#Build model
X = train[['Sex','Embarked','PassengerClass','SiblingOrSpouses','ParentsOrChildren']]
Xbar = test[['Sex','Embarked','PassengerClass','SiblingOrSpouses','ParentsOrChildren']]
y = train.Survived
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X,y)

#Predict
Ybar = decision_tree.predict(Xbar)
submission = pd.DataFrame({"PassengerId": test.index,"Survived": Ybar})

#Create CSV output
submission.to_csv('submission.csv', index=False)