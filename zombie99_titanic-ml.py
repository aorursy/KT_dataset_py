import numpy as np

import pandas as pd

import random as rnd

import os



import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

#import sklearn.linear_model as lm

%matplotlib inline



# machine learning

from numpy import nan

from sklearn.linear_model import LinearRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

print (os.listdir("../input"), "\n")



pd.set_option('display.expand_frame_repr', False)



data_train = pd.read_csv("../input/train.csv")

data_test = pd.read_csv("../input/test.csv")

data_train.head()

#print(data_train.columns.values, "\n")

desc = data_train.describe(include=['O'])

#print(desc, "\n")

byClass = data_train[['Pclass','Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by=['Survived'],ascending= False)

#print(byClass, "\n")

bySex = data_train[['Sex','Survived']].groupby(['Sex'], as_index=False).mean()

#print(bySex)

#g = sns.FacetGrid(data_train,col = 'Survived')

#g.map(plt.hist,'Age', bins=20)



#grid = sns.FacetGrid(data_train,col='Survived',row = 'Pclass', size=2.2, aspect=1.6)

#grid.map(plt.hist,'Age', alpha=.5,bins=20)



#grid_sex = sns.FacetGrid(data_train,col='Survived',row = 'Sex')

#grid_sex.map(plt.hist,'Age',bins=20)



data_train['Age'] = data_train['Age'].fillna(0).astype(int)

data_test['Age'] = data_test['Age'].fillna(0).astype(int)



data_train["Age"] = data_train["Age"].fillna(0);

data_train["Fare"] = data_train["Fare"].fillna(0);

data_test["Age"] = data_test["Age"].fillna(0);

data_test["Fare"] = data_test["Fare"].fillna(0);

#data_train = data_train[pd.notnull(data_train['Age'])]

#data_train = data_train[pd.notnull(data_train['Fare'])]

#data_test = data_test[pd.notnull(data_test['Age'])]

#data_test = data_test[pd.notnull(data_test['Fare'])]



data_testing = data_test



data_train = data_train.drop(['PassengerId','Ticket', 'Cabin','Embarked','Name','Parch','SibSp'], axis=1)

data_test = data_test.drop(['PassengerId','Ticket', 'Cabin','Embarked','Name','Parch','SibSp'], axis=1)



data_train['Sex'] = data_train['Sex'].map({'male': 0,'female': 1})

data_test['Sex'] = data_test['Sex'].map({'male': 0,'female': 1})



data_join = [data_train,data_test]





 

#telo = data_train[data_train.isna().any(axis=1)]

#telo_tes = data_test[data_test.isna().any(axis=1)]

 

#for dataset in data_join:

#   dataset['Sex'] =  dataset['Sex'].map( {'male': 0, 'female': 1} ).astype(int)



for dataset in data_join: 

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']



x_train = data_train.drop("Survived",axis=1)

y_train = data_train["Survived"]

x_test  = data_test

x_train.shape, y_train.shape, x_test.shape



#print(x_test)



#print("x=>",x_train,"\n")

#print("y=>",y_train,"\n")

    

#logreg = LogisticRegression()

#logreg.fit(x_train, y_train)

#y_pred = logreg.predict(x_test)

#acc_log = round(logreg.score(x_train, y_train) * 100, 2)

#acc_log



svc = SVC()

svc.fit(x_train, y_train)

y_pred = svc.predict(x_test)

acc_svc = round(svc.score(x_train, y_train) * 100, 2)

print(data_testing["PassengerId"],y_pred)

tello = pd.DataFrame({

        "PassengerId": data_testing["PassengerId"],

       "Survived": y_pred

    })

tello.head()

tello.to_csv('../submission.csv', index=False)