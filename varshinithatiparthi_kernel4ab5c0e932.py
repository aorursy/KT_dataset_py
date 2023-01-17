# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import sklearn

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/titanic/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
titanic_train = pd.read_csv('/kaggle/input/titanic/train.csv')

titanic_test = pd.read_csv('/kaggle/input/titanic/test.csv')

titanic_train

titanic_test
titanic_train.info()
titanic_train['Ticket'].value_counts()
titanic_train['Cabin'].value_counts()
titanic_train['Name'].value_counts()
titanic_train.isnull().sum()
titanic_test.isnull().sum()
import seaborn as sns

sns.set(style = "darkgrid")

sns.countplot(x = 'Survived', data =titanic_train)
titanic_train.describe()
titanic_data = titanic_train.drop(['Name','Ticket','Cabin'],axis = 1)

titanic_data
test_data = titanic_test.drop(['Name','Ticket','Cabin'],axis = 1)

test_data
titanic_train['Sex'].value_counts()


titanic_data['Sex'].replace(to_replace=["male","female"], value=[0,1],inplace=True)



test_data['Sex'].replace(to_replace=["male","female"], value=[0,1],inplace=True)
titanic_data.head()
test_data['Sex'].value_counts()
titanic_data['Embarked'].value_counts()
titanic_data['Embarked'].replace(to_replace=['S','C','Q'], value=[1,2,3],inplace=True)
test_data['Embarked'].replace(to_replace=['S','C','Q'], value=[1,2,3],inplace=True)
titanic_data.head()
titanic_data['Embarked'].value_counts()
titanic_data['Age'].value_counts()
titanic_data[["Age", "Parch"]].groupby(['Age'], as_index=False).mean().sort_values(by='Parch', ascending=False)
titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace = True)

titanic_data['Age'].value_counts()
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace = True)

titanic_data['Embarked'].value_counts()
test_data['Age'].fillna(test_data['Age'].median(), inplace = True)

test_data['Age'].value_counts()
test_data.fillna({'Fare':0}, inplace=True)

print(test_data)
test_data.isnull().sum()
titanic_data[["Age", "Survived"]].groupby(['Age'], as_index=False).mean().sort_values(by='Survived', ascending=False)
titanic_data['Embarked'].value_counts()
titanic_data.isnull().sum()



sns.set_style('whitegrid')

sns.pairplot(titanic_data)
y = titanic_data['Survived']

X = titanic_data[['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]

print('X: ', X.shape)

print('y:', y.shape)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state=4)

print("X_trian",X_train.shape)

print("y_train",y_train.shape)

print("X_test",X_test.shape)

print("y_test",y_test.shape)
clf =LogisticRegression(random_state=0)

clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
from sklearn.metrics import f1_score

f1_score(y_test,y_pred)
neigh = KNeighborsClassifier(n_neighbors=5)

neigh.fit(X_train,y_train)
y_predicted = neigh.predict(X_test)

f1_score(y_test,y_predicted)
clf = RandomForestClassifier(n_estimators=1000)

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

f1_score(y_test,y_pred)
test_survived = clf.predict(test_data)

test_survived
from sklearn.ensemble import ExtraTreesClassifier

tree = ExtraTreesClassifier(n_estimators=100)

tree.fit(X_train,y_train)

y_predicted = tree.predict(X_test)
f1_score(y_test,y_predicted)
test_survived = tree.predict(test_data)

test_survived
test_data['Survived'] = pd.DataFrame(test_survived)

test_data
submit = test_data[['PassengerId','Survived']]

submit.to_csv("/kaggle/working/submit.csv", index=False)