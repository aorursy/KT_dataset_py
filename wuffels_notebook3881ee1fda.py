# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import re as re

from subprocess import check_output

# print(check_output(["ls", "../input"]).decode("utf8"))
train_data = pd.read_csv("../input/train.csv")

print(train_data.info())

print
print (train_data.groupby(['Pclass'])['Pclass', 'Survived'].mean())
print (train_data.groupby(['Sex'])['Sex', 'Survived'].mean())
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1

print (train_data.groupby(['FamilySize'])['FamilySize', 'Survived'].mean())
train_data['IsAlone'] = 0

train_data.loc[train_data['FamilySize'] == 1, 'IsAlone'] = 1

print (train_data[['IsAlone', 'Survived']].groupby(['IsAlone']).mean())
print (train_data[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())
train_data['CategoricalFare'] = pd.qcut(train_data['Fare'], 5)

print (train_data[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean())
train_data['CategoricalFare'] = pd.cut(train_data['Fare'], 5)

print (train_data[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean())

print (train_data[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).count())
train_data['CategoricalAge'] = pd.cut(train_data['Age'], 5)

print (train_data[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean())

print (train_data[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).count())
def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    if title_search:

        return title_search.group(1)

    return ""



train_data['Title'] = train_data['Name'].apply(get_title)

print(train_data[['Title', 'Survived']].groupby(['Title']).mean())

print(pd.crosstab(train_data['Title'], train_data['Sex']))
train_data['Title'] = train_data['Title'].replace(['Lady', 'Countess','Capt', 'Col',

                                            'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],

                                            'Other')



train_data['Title'] = train_data['Title'].replace('Mlle', 'Miss')

train_data['Title'] = train_data['Title'].replace('Ms', 'Miss')

train_data['Title'] = train_data['Title'].replace('Mme', 'Mrs')



print (train_data[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())
train_data['HasAge'] = 1

train_data.loc[train_data['Age'].isnull(), 'HasAge'] = 0

print (train_data[['HasAge', 'Survived']].groupby(['HasAge']).mean())
train_data['SharedTicked'] = 0

grouped = train_data.groupby('Ticket')

k = 0

for name, group in grouped:

    g = grouped.get_group(name)

    if (len(g) > 1):

        train_data.loc[train_data['Ticket'] == name, 'SharedTicked'] = 1

        k += 1

print (train_data[['SharedTicked', 'Survived']].groupby(['SharedTicked']).mean())
train_data['Child'] = train_data['Age']<=10

print (train_data.groupby(['Child'])['Survived'].mean())

print("_____________________________")

tab = pd.crosstab(train_data['Child'], train_data['Pclass'])

print(tab)

print("_____________________________")

tab = pd.crosstab(train_data['Child'], train_data['Sex'])

print(tab)
print (train_data.groupby(['Ticket'])['Ticket', 'Survived'].mean())
train_data['IsAlone'] = 0

train_data.loc[train_data['FamilySize'] == 1, 'IsAlone'] = 1

print (train_data.groupby(['Ticket'])['Ticket', 'Survived'].mean())