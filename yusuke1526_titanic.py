# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import random as rnd
# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
combine = [train_df, test_df]

print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)



train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)

test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)

combine = [train_df, test_df]



print("After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

print(combine)
for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)

    

print(combine)

pd.crosstab(train_df['Title'], train_df['Sex'])
for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

    'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Rare':5}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)

    

train_df.head()
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)

test_df = test_df.drop(['Name'], axis=1)

combine = [train_df, test_df]
for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map({'female':0, 'male':1}).astype(int)

    

train_df.head()
guess_ages = np.zeros((2,3))
for dataset in combine:

    for i in range(0, 2):

        for j in range(0, 3):

            guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()

            #age_guess = guess_df.median()

            #guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5



            guess_ages[i,j] = guess_df.median()



    for i in range(0, 2):

        for j in range(0, 3):

            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),'Age'] = guess_ages[i,j]



    dataset['Age'] = dataset['Age'].astype(int)



train_df.head()
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)

train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
for dataset in combine:

    dataset.loc[dataset['Age']<=16, 'Age'] = 0

    dataset.loc[(dataset['Age']>16) & (dataset['Age']<=32), 'Age'] = 1

    dataset.loc[(dataset['Age']>32) & (dataset['Age']<=48), 'Age'] = 2

    dataset.loc[(dataset['Age']>48) & (dataset['Age']<=64), 'Age'] = 3

    dataset.loc[dataset['Age']>64, 'Age'] = 4

train_df.head()
train_df = train_df.drop(['AgeBand'], axis=1)

combine = [train_df, test_df]

train_df.head()
for dataset in combine:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    

train_df[['FamilySize','Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize']==1, 'IsAlone'] = 1

    

train_df[['IsAlone','Survived']].groupby(['IsAlone'], as_index=False).mean()
train_df = train_df.drop(['Parch','SibSp','FamilySize'], axis=1)

test_df = test_df.drop(['Parch','SibSp','FamilySize'], axis=1)

combine = [train_df, test_df]
freq_port = train_df.Embarked.dropna().mode()[0]

freq_port
for dataset in combine:

    print(dataset[820:840]['Embarked'])



print(train_df[829:830])

train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)



train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df.info()
for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)

    

train_df.head()
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].dropna().median())

test_df.info()
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)

train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
for dataset in combine:

    dataset.loc[dataset['Fare']<=7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare']>7.91) & (dataset['Fare']<=14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare']>14.454) & (dataset['Fare']<=31.0), 'Fare'] = 2

    dataset.loc[dataset['Fare']>31.0, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)

    

train_df = train_df.drop(['FareBand'], axis=1)

combine = [train_df, test_df]



train_df.head(10)
test_df
train_df = train_df.drop(['IsAlone'], axis=1)

test_df = test_df.drop(['IsAlone'], axis=1)
X_train = train_df.drop(['Survived'], axis=1)

Y_train = train_df['Survived']

X_test = test_df.drop(['PassengerId'], axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log
coeff_df = pd.DataFrame(train_df.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df['Correlation'] = pd.Series(logreg.coef_[0])



coeff_df.sort_values(by='Correlation', ascending=False)
from sklearn.model_selection import cross_val_score

random_forest = RandomForestClassifier(n_estimators=100)

# cross_validation

scores = cross_val_score(random_forest, X_train, Y_train, cv=3)



print('Cross-Validation scores: {}'.format(scores))



print('Average score: {}'.format(np.mean(scores)))
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)
coeff_df = pd.DataFrame(train_df.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df['Correlation'] = pd.Series(random_forest.feature_importances_)



coeff_df.sort_values(by='Correlation', ascending=False)
submission = pd.DataFrame({

    'PassengerId': test_df['PassengerId'],

    'Survived': Y_pred

})
submission.to_csv('submission.csv', index=False)