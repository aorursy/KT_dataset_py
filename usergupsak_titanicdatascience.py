# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
raw_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

total = [raw_data, test_data]

raw_data.head()
raw_data.describe()
raw_data.info()
raw_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index = False).mean()
raw_data[['Sex', 'Survived']].groupby(['Sex'], as_index = False).mean()
raw_data[['SibSp', 'Survived']].groupby(['SibSp'], as_index = False).mean()
raw_data[['Parch', 'Survived']].groupby(['Parch'], as_index = False).mean()
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



age_dist = sns.FacetGrid(raw_data, col = 'Survived')

age_dist.map(plt.hist, 'Age', bins = 100)
print('Before: ', raw_data.shape, test_data.shape)

raw_data = raw_data.drop(['Ticket', 'Cabin'], axis = 1)

test_data = test_data.drop(['Ticket', 'Cabin'], axis = 1)

print('After removing: ', raw_data.shape, test_data.shape)



total = [raw_data, test_data]
for dataset in total:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(raw_data['Title'], raw_data['Sex'])
for dataset in total:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

raw_data[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in total:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)



raw_data = raw_data.drop(['Name', 'PassengerId'], axis = 1)

test_data = test_data.drop(['Name'], axis = 1)

total = [raw_data, test_data]
raw_data.head()
for dataset in total:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
raw_data.head()
age_estimate = np.zeros((2,3))



for dataset in total:

    for i in range(0, 2):

        for j in range(1, 4):

            temp = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j)]['Age'].dropna()

            age_estimate[i, j - 1] = int(temp.median()/0.5 + 0.5) * 0.5



    for i in range(0, 2):

        for j in range(0, 3):

            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1), 'Age'] = age_estimate[i, j]

    dataset['Age'] = dataset['Age'].astype(int)

    

raw_data.head()
print('Data imputation done!')

print('-'*40)

raw_data.info()
raw_data['Age range'] = pd.cut(raw_data['Age'], 5)

raw_data[['Age range', 'Survived']].groupby(['Age range'], as_index = False).mean().sort_values(by = 'Age range', ascending = True)
for dataset in total:    

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4



raw_data = raw_data.drop(['Age range'], axis = 1)

total = [raw_data, test_data]

raw_data.head()
for dataset in total:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1



raw_data[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)
for dataset in total:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1



raw_data[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index = False).mean()
raw_data = raw_data.drop(['Parch', 'SibSp', 'FamilySize'], axis = 1)

test_data = test_data.drop(['Parch', 'SibSp', 'FamilySize'], axis = 1)

total = [raw_data, test_data]



raw_data.head()
port = raw_data.Embarked.dropna().mode()[0]



for dataset in total:

    dataset['Embarked'] = dataset['Embarked'].fillna(port)

    

raw_data[['Embarked', 'Survived']].groupby(['Embarked'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)
for dataset in total:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



raw_data['Embarked']
test_data['Fare'].fillna(test_data['Fare'].dropna().median(), inplace = True)
test_data.info()
raw_data.info()
raw_data['FareBand'] = pd.cut(raw_data['Fare'], 4)

raw_data[['FareBand', 'Survived']].groupby(['FareBand'], as_index = False).mean().sort_values(by = 'FareBand', ascending = True)
for dataset in total:

    dataset.loc[ dataset['Fare'] <= 128.082, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 128.082) & (dataset['Fare'] <= 256.165), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 256.165) & (dataset['Fare'] <= 384.247), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 384.247, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)



raw_data = raw_data.drop(['FareBand'], axis = 1)

total = [raw_data, test_data]

    

raw_data.head()
test_data.head()
X_train = raw_data.drop('Survived', axis = 1)

Y_train = raw_data['Survived']

X_test  = test_data.drop('PassengerId', axis = 1).copy()

X_train.shape, Y_train.shape, X_test.shape
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
###Model Testing



logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

accuracy_logreg = round(logreg.score(X_train, Y_train) * 100, 2)

print('Logistic Regression: ', accuracy_logreg)
features = pd.DataFrame(raw_data.columns.delete(0))

features.columns = ['Feature']

features["Correlation"] = pd.Series(logreg.coef_[0])

features.sort_values(by = 'Correlation', ascending = False)
raw_data.columns
logreg.coef_[0]
svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

accuracy_svc = round(svc.score(X_train, Y_train) * 100, 2)

accuracy_svc
knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

accuracy_knn = round(knn.score(X_train, Y_train) * 100, 2)

accuracy_knn
decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

accuracy_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

accuracy_tree
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

accuracy_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

accuracy_forest
submission = pd.DataFrame({"PassengerId": test_data["PassengerId"], "Survived": Y_pred})

submission.to_csv('output.csv', index = False)       