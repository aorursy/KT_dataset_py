# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Data Analysis 

import pandas as pd

import numpy as np

import random as rnd



# Visualization

import matplotlib.pyplot as plt

import seaborn as sns
train_df = pd.read_csv("/kaggle/input/titanic/train.csv")

test_df = pd.read_csv("/kaggle/input/titanic/test.csv")



total = [train_df, test_df]
print(train_df.columns.values)
train_df.head()
test_df.head()
train_df.tail()
test_df.tail()
train_df.info()
test_df.info()
train_df.describe()
train_df.describe(include=['O'])
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived',ascending= False)
train_df[['Sex','Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Sex',ascending=False)
train_df[['SibSp','Survived']].groupby(["SibSp"], as_index = False).mean().sort_values(by='SibSp', ascending=False)
train_df[['Parch', 'Survived']].groupby(['Parch'],as_index=False).mean().sort_values(by='Parch', ascending=False)
g1 = sns.FacetGrid(train_df, col='Survived', height=3, aspect=1.2)

g1.map(plt.hist, 'Age', bins=20)
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', height=3, aspect=1.5)

grid.map(plt.hist, "Age", alpha=0.5, bins=20)

grid.add_legend();
grid = sns.FacetGrid(train_df, row='Embarked', height=3, aspect=1.5)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived',height=3, aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha= 0.5, ci = None)

grid.add_legend()
print("before", train_df.shape, test_df.shape, total[0].shape, total[1].shape)



train_df.drop(['Ticket', 'Cabin'], axis=1, inplace=True)

test_df.drop(['Ticket', 'Cabin'], axis=1, inplace=True)



total=[train_df, test_df]



print("After", train_df.shape, test_df.shape, total[0].shape, total[1].shape)
for dataset in total:

    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.',expand = True)

    

pd.crosstab(train_df['Title'], train_df['Sex'])
for dataset in total:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don',\

                                                 'Dr', 'Major','Rev','Sir', 'Jonkheer', 'Dona'], "Rare")

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms','Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {"Mr":1, "Miss":2, "Mrs":3, "Master":4, "Rare":5}

for dataset in total:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)

    

train_df.head()
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)

test_df = test_df.drop(['Name'], axis=1)

total = [train_df, test_df]

train_df.shape, test_df.shape
for dataset in total:

    dataset['Sex'] = dataset['Sex'].map({'female':1, 'male':0}).astype(int)



train_df.head()
grid = sns.FacetGrid(train_df, row="Pclass", col="Sex",height=3, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=0.6, bins=20)

grid.add_legend()
guess_ages=np.zeros((2,3))

guess_ages
for dataset in total:

    for i in range(0, 2):

        for j in range(0, 3):

            guess_df = dataset[(dataset['Sex'] == i) & \

                                  (dataset['Pclass'] == j+1)]['Age'].dropna()



            age_guess = guess_df.median()



            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

            

    for i in range(0, 2):

        for j in range(0, 3):

            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\

                    'Age'] = guess_ages[i,j]



    dataset['Age'] = dataset['Age'].astype(int)



train_df.head()
train_df['Ageband'] = pd.cut(train_df['Age'], 5)

train_df[['Ageband', 'Survived']].groupby(['Ageband'], as_index=False).mean().sort_values(by='Ageband', ascending=True)
for dataset in total:

    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset["Age"] > 16) & (dataset["Age"] <=32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <=48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[dataset['Age'] > 64, 'Age']



train_df.head()
train_df = train_df.drop('Ageband', axis = 1)

total = [train_df, test_df]

train_df.head()
for dataset in total:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    

train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index= False).mean().sort_values(by='Survived', ascending=False)
for dataset in total:

    dataset["IsAlone"] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

    

train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
train_df = train_df.drop(['Parch', 'SibSp','FamilySize'], axis = 1)

test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis = 1)



total = [train_df, test_df]



train_df.head()
for dataset in total:

    dataset['Age*Class'] = dataset.Age * dataset.Pclass



train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)
freq_port = train_df.Embarked.dropna().mode()[0]

freq_port
for dataset in total:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)



train_df[['Embarked','Survived']].groupby(['Embarked'],as_index=False).mean().sort_values(by='Survived',ascending=False)
for dataset in total:

    dataset['Embarked'] =dataset['Embarked'].map({'S':0,'C':1,'Q':2}).astype(int)

    

train_df.head()
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)



test_df.head()
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)



train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending= False)
for dataset in total:

    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31.0), 'Fare'] = 2

    dataset.loc[dataset['Fare'] > 31.0, 'Fare'] = 3



    dataset['Fare'] = dataset['Fare'].astype(int)

    

train_df = train_df.drop(['FareBand'], axis =1)



total = [train_df, test_df]



train_df.head(10)
test_df.head(10)
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

X = train_df.drop("Survived", axis = 1)

y = train_df['Survived']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

X_test = test_df.drop("PassengerId", axis=1)

X_train.shape, X_valid.shape, X_test.shape, y_train.shape, y_valid.shape
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X_train,y_train)

y_pred = logreg.predict(X_valid)

logreg.fit(X, y)

accuracy_score(y_pred, y_valid)
coeff_df = pd.DataFrame(X.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(logreg.coef_[0])



coeff_df.sort_values(by='Correlation', ascending=False)
from sklearn.svm import SVC, LinearSVC

svc = SVC()

svc.fit(X_train,y_train)

y_pred = svc.predict(X_valid)

svc.fit(X_train, y_train)

accuracy_score(y_pred, y_valid)
linear_svc = LinearSVC()

linear_svc.fit(X_train,y_train)

y_pred = linear_svc.predict(X_valid)

linear_svc.fit(X,y)

accuracy_score(y_pred, y_valid)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 10)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_valid)

knn.fit(X_train,y_train)

accuracy_score(y_pred, y_valid)
from sklearn.naive_bayes import GaussianNB

gaussian = GaussianNB()

gaussian.fit(X_train, y_train)

y_pred = gaussian.predict(X_valid)

gaussian.fit(X,y)

accuracy_score(y_pred, y_valid)
from sklearn.linear_model import Perceptron

perceptron = Perceptron()

perceptron.fit(X_train,y_train)

y_pred = perceptron.predict(X_valid)

perceptron.fit(X,y)

accuracy_score(y_pred,y_valid)
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()

sgd.fit(X_train, y_train)

y_pred = sgd.predict(X_valid)

sgd.fit(X,y)

accuracy_score(y_pred, y_valid)
from sklearn.tree import DecisionTreeClassifier

DTC = DecisionTreeClassifier()

DTC.fit(X_train,y_train)

y_pred = DTC.predict(X_valid)

DTC.fit(X,y)

accuracy_score(y_pred, y_valid)
from sklearn.ensemble import RandomForestClassifier 

RFC = RandomForestClassifier(n_estimators=200)

RFC.fit(X_train,y_train)

y_pred = RFC.predict(X_valid)

RFC.fit(X,y)

accuracy_score(y_pred, y_valid)
final_pred = RFC.predict(X_test)

final_pred
Submit = pd.DataFrame({'PassengerId':test_df['PassengerId'], 'Survived':final_pred })
Submit.to_csv('submission.csv', index=False)