# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

import re

print(check_output(["ls", "../input"]).decode("utf8"))



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



full = pd.concat([train,test], ignore_index=True)

full.head()





total = train.isnull().sum().sort_values(ascending = False)

p1 = train.isnull().sum()/ train.isnull().count()*100

p2 = (round(p1, 1)).sort_values(ascending = False)

missingdata = pd.concat([total, p2], axis = 1, keys = ['Total', '%'])

missingdata.head()



train.columns.values

survived = 'survived'

not_survived = 'not survived'

 

fig, axes = plt.subplots(nrows=1, ncols = 2,figsize=(10,4))

women = train[train['Sex'] == 'female']

men = train[train['Sex'] == 'male']

ax = sns.distplot(women[women['Survived'] == 1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde = False)

ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde = False)

ax.legend()

ax.set_title('Female')

ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False)

ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False)

ax.legend()

_ = ax.set_title('Male')
FacetGrid = sns.FacetGrid(train, row = 'Embarked', size = 5, aspect = 1.6)

FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette=None, order=None,hue_order=None)

FacetGrid.add_legend()
sns.barplot(x = 'Pclass', y = 'Survived', data = train)
grid = sns.FacetGrid(train, col='Survived', row = 'Pclass', size = 5, aspect = 1.6)

grid.map(plt.hist, 'Age', alpha = 0.5, bins = 20)

grid.add_legend()
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index = False).mean().sort_values(by = 'Survived', ascending=False)

train[['Sex', 'Survived']].groupby(['Sex'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)

train[['SibSp', 'Survived']].groupby(['SibSp'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)

train = train.drop(['PassengerId'], axis = 1)
train.head()
data = [train, test]

title = {"Mr." : 1, "Miss" : 2, "Mrs." : 3, "Master" : 4, "Rare" : 5}



for dataset in data:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.')

    dataset['Title']=dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr','Major', 'Rev','Sir','Dona'] , 'Rare')

    dataset['Title'] = dataset['Title'].map(title)

    dataset['Title'] = dataset['Title'].fillna(0)

    dataset['Title'] = dataset['Title'].astype(int)



dataset.head()

train = train.drop(['Name'], axis = 1)

test = test.drop(['Name'], axis = 1)
# Convert Sex feature in Numeric



gender = {'male' : 0, "female" : 1}

data = [train, test]

for dataset in data:

    dataset['Sex'] = dataset['Sex'].map(gender)
train['Ticket']. describe()
train = train.drop(['Ticket'], axis =1)
test = test.drop(['Ticket'], axis = 1)
# Now convert Embarked into Numeric



common = 'S'

port = {'S' : 0, 'C' : 1, 'Q' : 2}

data = [train, test]



for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].fillna(common)

    dataset['Embarked'] = dataset['Embarked'].map(port)
data = [train, test]

for dataset in data:

    dataset['Age'] = dataset['Age'].fillna(0)

    dataset['Age'] = dataset['Age'].astype(int)

    dataset.loc[(dataset['Age'] <= 11), 'Age'] = 0

    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3

    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4

    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5

    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6

    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6
test.head()
# Categorize based on Fare

data = [train, test]

for dataset in data:

    dataset['Fare'] = dataset['Fare'].fillna(0)

    dataset.loc[dataset['Fare'] <= 7.98, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.98) & (dataset['Fare'] <= 15.01), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 15.01) & (dataset['Fare'] <= 30.11), 'Fare'] = 2

    dataset.loc[(dataset['Fare'] > 30.11) & (dataset['Fare'] <= 100.02), 'Fare'] = 3

    dataset.loc[(dataset['Fare'] > 100.02) & (dataset['Fare'] <= 250.12), 'Fare'] = 4

    dataset.loc[ dataset['Fare'] > 250.12, 'Fare'] = 5

    

    dataset['Fare'] = dataset['Fare'].astype(int)

    
train.head()
# Find Fare per Person

data = [train, test]

for dataset in data:

    dataset['Relative'] = dataset['SibSp'] + dataset['Parch']

    dataset.loc[dataset['Relative'] > 0, 'not_alone'] = 0

    dataset.loc[dataset['Relative'] == 0, 'not_alone'] = 1

    dataset['not_alone'] = dataset['not_alone'].astype(int)

    dataset['FarePerPerson'] = dataset['Fare']/(dataset['Relative'] + 1)

deck = { 'A' : 1, 'B' : 2,'C' : 3, 'D':4,'E':5,'F':6,'G':7,'U':8}

data = [train, test]

for dataset in data:

    dataset['Cabin'] = dataset['Cabin'].fillna('U0')

    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([A-Za-z]+)").search(x).group())

    dataset['Deck'] = dataset['Deck'].map(deck)

    dataset['Deck'] = dataset['Deck'].fillna(0)

    dataset['Deck'] = dataset['Deck'].astype(int)

    

train = train.drop(['Cabin'], axis = 1)

test = test.drop(['Cabin'], axis = 1)
train.head()
from sklearn.neighbors import KNeighborsClassifier

test.head()
X_train = train.drop('Survived', axis =1)

Y_train = train['Survived']

X_test = test.drop('PassengerId', axis=1).copy()
X_train
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_prediction = knn.predict(X_test)

accuracy = knn.score(X_train, Y_train)
accuracy