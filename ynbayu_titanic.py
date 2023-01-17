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
import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test_d = pd.read_csv('/kaggle/input/titanic/test.csv')
train = train.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin'], axis=1)
test = test_d.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin'], axis=1)
train['Age'].fillna(train['Age'].mean(), inplace=True)
test['Age'].fillna(test['Age'].mean(), inplace=True)
test['Fare'].fillna(test['Fare'].mean(), inplace=True)
train['Embarked'].fillna('S', inplace=True)
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

train['Sex'] = le.fit_transform(train['Sex'])

test['Sex'] = le.transform(test['Sex'])
train['Age'] = pd.cut(train['Age'], bins=6)

test['Age'] = pd.cut(test['Age'], bins=6)
train['Fare'] = pd.cut(train['Fare'], bins=12)

test['Fare'] = pd.cut(test['Fare'], bins=12)
train['Fare'] = le.fit_transform(train['Fare'])

test['Fare'] = le.transform(test['Fare'])
cols_to_lb = ['Age','Embarked']
train['Age'] = le.fit_transform(train['Age'])

test['Age'] = le.fit_transform(test['Age'])
train['Embarked'] = le.fit_transform(train['Embarked'])

test['Embarked'] = le.transform(test['Embarked'])
test.head()
train.head()
y = train['Survived']

X = train.drop(['Survived'], axis=1)
X.head()
y.head()
from sklearn.model_selection import train_test_split



X_train, X_valid, y_train, y_valid = train_test_split(X,y,random_state=42)
X_train
y_train
X_valid
y_valid
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score
def score_t(n):

    xgb_clf = XGBClassifier(n_estimators = n, learning_rate = 0.001)

    xgb_clf.fit(X_train,y_train)

    train_predict = xgb_clf.predict(X_train)

    print(accuracy_score(train_predict,y_train))
for i in range(50,1001,50):

    print(score_t(i))
def score(n):

    xgb_clf = XGBClassifier(n_estimators = n, learning_rate = 0.001)

    xgb_clf.fit(X_train,y_train)

    valid_predict = xgb_clf.predict(X_valid)

    print(accuracy_score(valid_predict,y_valid))
for i in range(50,1001,50):

    print(score(i))
xgb_clf = XGBClassifier(n_estimators = 100, learning_rate = 0.001)

xgb_clf.fit(X_train,y_train)

test_predict = xgb_clf.predict(test)

test_predict
output = pd.DataFrame({'PassengerId':test_d['PassengerId'],

                       'Survived':test_predict})

output
output.to_csv('titanic_sub.csv',index=False)