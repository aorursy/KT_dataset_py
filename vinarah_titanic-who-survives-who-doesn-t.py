import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

full_df = [train_df, test_df]
train_df.info()
train_df.head()
sns.heatmap(train_df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
sns.set_style('whitegrid')

sns.countplot(x='Survived', data=train_df)
sns.countplot(x='Survived', hue='Sex', data=train_df, palette='RdBu_r')
train_df[['Survived', 'Sex']].groupby(['Sex'], as_index=False).mean()
sns.countplot(x='Survived', hue='Pclass', data=train_df)
train_df[['Survived', 'Pclass']].groupby(['Pclass'], as_index=False).mean()
for data_frame in full_df:

    data_frame['FamilySize'] = data_frame['SibSp'] + data_frame['Parch'] + 1
train_df[['Survived', 'FamilySize']].groupby(['FamilySize'], as_index=False).mean()
def check_if_alone(cols):

    size = cols[0]

    if size == 1:

        return 1

    else:

        return 0



for data_frame in full_df:

    data_frame['IsAlone'] = data_frame[['FamilySize']].apply(check_if_alone, axis=1)
train_df[['Survived', 'IsAlone']].groupby(['IsAlone'], as_index=False).mean()
for data_frame in full_df:

    data_frame['Embarked'] = data_frame['Embarked'].fillna('Q')
sns.countplot(x='Survived', hue='Embarked', data=train_df)
train_df[['Survived', 'Embarked']].groupby(['Embarked'], as_index=False).mean()
for data_frame in full_df:

    data_frame['Fare'] = data_frame['Fare'].fillna(train_df['Fare'].median())

train_df['CategoricalFare'] = pd.qcut(train_df['Fare'], 4)

train_df[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean()
def cabin_known(cabin):

    if pd.isnull(cabin):

        return 0

    else:

        return 1



for data_frame in full_df:

    data_frame['CabinKnown'] = data_frame['Cabin'].apply(cabin_known)
train_df[['CabinKnown', 'Survived']].groupby(['CabinKnown'], as_index=False).mean()
train_df[['Age', 'Pclass']].groupby(['Pclass'], as_index=False).mean()
def impute_age(cols):

    age = cols[0]

    cls = cols[1]

    if pd.isnull(age):

        if cls == 1:

            return 38

        elif cls == 2:

            return 29

        else:

            return 25

    else:

        return age



for data_frame in full_df:

    data_frame['Age'] = data_frame[['Age', 'Pclass']].apply(impute_age,axis=1)
train_df['CategoricalAge'] = pd.cut(train_df['Age'],5)

train_df[['CategoricalAge','Survived']].groupby(['CategoricalAge'], as_index=False).mean()
import re as re

def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.',name)

    if title_search:

        return title_search.group(1)

    return ''

for data_frame in full_df:

    data_frame['Title'] = data_frame['Name'].apply(get_title)
pd.crosstab(train_df['Title'], train_df['Sex'])
for data_frame in full_df:

    data_frame['Title'] = data_frame['Title'].replace(['Lady','Countess', 'Col','Don', 'Dr', 'Major',

            'Rev', 'Sir', 'Jonkheer', 'Dona','Capt'],'Rare')

    data_frame['Title'] = data_frame['Title'].replace('Mlle', 'Miss')

    data_frame['Title'] = data_frame['Title'].replace('Ms', 'Miss')

    data_frame['Title'] = data_frame['Title'].replace('Mme', 'Mrs')
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
def map_fare(fare):

    if fare <= 7.91:

        return 0

    elif fare <= 14.454:

        return 1

    elif fare <= 31:

        return 2

    else:

        return 3

    

for data_frame in full_df:

    data_frame['Fare'] = data_frame['Fare'].apply(map_fare)
def map_age(age):

    if age <=16:

        return 0

    elif age <=32:

        return 1

    elif age <=48:

        return 2

    elif age <=64:

        return 3

    else:

        return 4



for data_frame in full_df:

    data_frame['Age'] = data_frame['Age'].apply(map_age)
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for data_frame in full_df:

    data_frame['Title'] = data_frame['Title'].map(title_mapping)

    data_frame['Title'] = data_frame['Title'].fillna(0)
sex = pd.get_dummies(train_df['Sex'],drop_first=True)

embark = pd.get_dummies(train_df['Embarked'],drop_first=True)

train_df = pd.concat([train_df, sex, embark], axis=1)
sex = pd.get_dummies(test_df['Sex'],drop_first=True)

embark = pd.get_dummies(test_df['Embarked'],drop_first=True)

test_df = pd.concat([test_df, sex, embark], axis=1)
ids = test_df['PassengerId'].values

drop_elements = ['PassengerId', 'Name','Sex','Embarked', 'Ticket', 'Cabin', 'SibSp','Parch', 'FamilySize']

train_df = train_df.drop(drop_elements, axis=1)

test_df = test_df.drop(drop_elements, axis=1)

train_df = train_df.drop(['CategoricalFare','CategoricalAge'],axis=1)
train_df.head()
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF







X_train, X_test, y_train, y_test = train_test_split(train_df.drop(['Survived'], axis=1), train_df['Survived'], test_size=0.3, random_state=101)



classifier = GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True)

classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)
print(classification_report(y_test,predictions))
classifier.fit(train_df.drop(['Survived'], axis=1), train_df['Survived'])
result = classifier.predict(test_df)