# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random as rnd



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import warnings

warnings.filterwarnings("ignore")



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

combine = [train,test]
print(train.columns.values)
train.head()
train.info()

print('*'*50)

test.info()
train.describe()
train.describe(include=[np.object])

#train.describe(include=['O'])
train[['Pclass','Survived']].groupby(['Pclass'],as_index=False)['Survived'].mean()
train[['Sex','Survived']].groupby(['Sex'],as_index=False)['Survived'].mean()
train[['SibSp','Survived']].groupby(['SibSp'],as_index=False)['Survived'].mean().sort_values(

    by='Survived',ascending=False)
train[["Parch", "Survived"]].groupby(['Parch'], as_index=False)['Survived'].mean().sort_values(

    by='Survived', ascending=False)
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
g = sns.FacetGrid(train,col='Survived')

g.map(plt.hist,'Age',bins=20)
grid = sns.FacetGrid(train,col='Survived',row='Pclass',size=3,aspect=1.5)

grid.map(plt.hist,'Age',alpha=.5,bins=20)

grid.add_legend()
grid = sns.FacetGrid(train, row='Embarked', size=2.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex')

grid.add_legend()
grid = sns.FacetGrid(train,row='Embarked',col='Survived')

grid.map(sns.barplot, 'Sex', 'Fare', alpha=.3, ci=None)

grid.add_legend()
train['Cabin'].fillna('U0',inplace = True)

test['Cabin'].fillna('U0',inplace = True)

#train[train['Cabin'] == 'U0']
grid = sns.FacetGrid(train[train['Cabin'] == 'U0'])

grid.map(plt.hist,'Survived',color='r')

grid = sns.FacetGrid(train[train['Cabin'] != 'U0'])

grid.map(plt.hist,'Survived',color='b')
train = train.drop(['Ticket','PassengerId'],axis=1)

test = test.drop(['Ticket','PassengerId'],axis=1)

combine = [train,test]

for dataset in combine:

    dataset['Title'] = dataset['Name'].str.extract('([A-Za-z]+)\.',expand=False)

pd.crosstab(train['Title'],train['Sex'])
for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

train[['Title','Survived']].groupby(['Title'],as_index=False).mean().sort_values(by='Survived')
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)

train = train.drop(['Name'],axis=1)

test = test.drop(['Name'],axis=1)

combine = [train,test]
train.head()
for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map({'female':1, 'male':0}).astype(int)

train.head()
grid = sns.FacetGrid(train,row='Pclass',col='Sex',size=3,aspect=1.6)

grid.map(plt.hist,'Age',bins=25)

grid.add_legend()
guess_age = np.zeros((2,3))

guess_age
for dataset in combine:

    for i in range(0,2):

        for j in range(0,3):

            guess = dataset[(dataset['Sex']==i) & (dataset['Pclass']==j+1)]['Age'].dropna()

            age_guess = guess.median()

            guess_age[i,j] = int(age_guess/0.5 + 0.5)*0.5

    for i in range(0, 2):

        for j in range(0, 3):

            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),'Age'] = guess_age[i,j]



    dataset['Age'] = dataset['Age'].astype(int)

train.head()
train['AgeBand'] = pd.cut(train['Age'],5)

train[['AgeBand','Survived']].groupby(['AgeBand'],as_index=False).mean().sort_values(by='AgeBand', ascending=True)
for dataset in combine:

    dataset.loc[dataset['Age']<=16,'Age'] = 0

    dataset.loc[(dataset['Age']<=32) & (dataset['Age']>16),'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']

train = train.drop(['AgeBand'],axis=1)

combine = [train,test]
for dataset in combine:

    dataset['Age*Class'] = dataset.Age * dataset.Pclass
for dataset in combine:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train[['FamilySize','Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1



train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
for dataset in combine:

    dataset['HasCabin'] = 1

    dataset.loc[dataset['Cabin'] == 'U0', 'HasCabin'] = 0



train[['HasCabin', 'Survived']].groupby(['HasCabin'], as_index=False).mean()
train = train.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

test = test.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

combine = [train,test]

train
train['FareBand'] = pd.qcut(train['Fare'],4)

train[['FareBand','Survived']].groupby(['FareBand'],as_index=False).mean().sort_values(by='FareBand', ascending=True)
test['Fare'].fillna(test['Fare'].dropna().median(),inplace=True)
for dataset in combine:

    dataset.loc[dataset['Fare']<=7.91,'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)



train = train.drop(['FareBand'], axis=1)

combine = [train,test]
freq_port = train.Embarked.mode()[0]

for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
train = train.drop(['Cabin'], axis=1)

test = test.drop(['Cabin'], axis=1)

train = pd.get_dummies(train)

test = pd.get_dummies(test)

combine = [train,test]
train
test
train_x = train.drop('Survived', axis=1)

train_y = train['Survived']

test_x = test

train_x.shape,train_y.shape,test_x.shape
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

random_forest = RandomForestClassifier(n_estimators=100)



results = cross_val_score(random_forest,train_x,train_y,scoring='accuracy',cv=10)

print(np.mean(results))
random_forest.fit(train_x,train_y)

predict = random_forest.predict(test_x)

col = pd.read_csv('../input/test.csv')

output = pd.DataFrame({'PassengerId':col['PassengerId'],'Survived':predict})

output.to_csv('submission.csv',index=False)