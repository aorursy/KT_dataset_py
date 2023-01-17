import numpy as np 

import pandas as pd 

#import os

import matplotlib.pyplot as plt

import seaborn as sns

from keras.layers import Input, Dense, BatchNormalization, Add, GaussianNoise, Dropout

from keras.models import Model

from keras import regularizers



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.preprocessing import StandardScaler



import warnings

warnings.filterwarnings("ignore")



sns.set() 

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

print ("Train: ",train.shape)

print ("Test: ",train.shape)


train.head()

train.isnull().sum()

sns.barplot(x='Pclass', y='Survived', data=train)
sns.barplot(x='Embarked', y='Survived', data=train)
sns.barplot(x='Sex', y='Survived', data=train)
sns.barplot(x='Survived', y='Fare', data=train)
train_test_data = [train, test]

for dataset in train_test_data:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.')

    
for dataset in train_test_data:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', \

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5}

for dataset in train_test_data:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)
train.head()
for dataset in train_test_data:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
train.head()
for dataset in train_test_data:

    age_avg = dataset['Age'].mean()

    age_std = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)

    

train['AgeBand'] = pd.cut(train['Age'], 5)



print (train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean())
for dataset in train_test_data:

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
train.head()
for dataset in train_test_data:

    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
train['FareBand'] = pd.qcut(train['Fare'], 4)

print (train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean())
for dataset in train_test_data:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)
train.head()
for dataset in train_test_data:

    dataset['FamilySize'] = dataset['SibSp'] +  dataset['Parch'] + 1



print (train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())
for dataset in train_test_data:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

    

print (train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())
embark_dummies_titanic  = pd.get_dummies(train['Embarked'])

embark_dummies_titanic.drop(['S'], axis=1, inplace=True)



embark_dummies_test  = pd.get_dummies(test['Embarked'])

embark_dummies_test.drop(['S'], axis=1, inplace=True)



train = train.join(embark_dummies_titanic)

test  = test.join(embark_dummies_test)



train.drop(['Embarked'], axis=1,inplace=True)

test.drop(['Embarked'], axis=1,inplace=True)
features_drop = ['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin']

train = train.drop(features_drop, axis=1)

test = test.drop(features_drop, axis=1)

train = train.drop(['PassengerId', 'AgeBand', 'FareBand'], axis=1)
train.head()
test.head()
X_train = train.drop('Survived', axis=1)

y_train = train['Survived']

X_test = test.drop("PassengerId", axis=1).copy()



X_train.shape, y_train.shape, X_test.shape
#Se debe realizar ONE HOT ENCODING ANTES DE NORMALIZAR
#sc = StandardScaler()

#X_train = sc.fit_transform(X_train)

#X_test = sc.transform(X_test)
logreg = LogisticRegression()

logreg.fit(X_train, y_train)

Y_pred = logreg.predict(X_test)

logreg.score(X_train, y_train)
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, y_train)

y_pred = random_forest.predict(X_test)

random_forest.score(X_train, y_train)
