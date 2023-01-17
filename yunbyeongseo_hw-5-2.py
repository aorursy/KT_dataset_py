# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn import tree

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")

gender_data = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")

train_data.info()
#성별을 숫자화 함

train['Sex_clean'] = train['Sex'].astype('category').cat.codes

test['Sex_clean'] = test['Sex'].astype('category').cat.codes

#어느 항구에서 타는지 값 처리 

train['Embarked'].fillna('S', inplace=True)

train['Embarked_clean'] = train['Embarked'].astype('category').cat.codes

test['Embarked_clean'] = test['Embarked'].astype('category').cat.codes

#가족

train['Family'] = 1 + train['SibSp'] + train['Parch']

test['Family'] = 1 + test['SibSp'] + test['Parch']

#혼자탔는지

train['Solo'] = (train['Family'] == 1)

test['Solo'] = (test['Family'] == 1)

#fare

train['FareBin'] = pd.qcut(train['Fare'], 5)

test['FareBin'] = pd.qcut(test['Fare'], 5)

train['Fare_clean'] = train['FareBin'].astype('category').cat.codes

test['Fare_clean'] = test['FareBin'].astype('category').cat.codes

#title

train['Title'] = train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

test['Title'] = test['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

train['Title'] = train['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')

train['Title'] = train['Title'].replace('Mlle', 'Miss')

train['Title'] = train['Title'].replace('Ms', 'Miss')

train['Title'] = train['Title'].replace('Mme', 'Mrs')

test['Title'] = test['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')

test['Title'] = test['Title'].replace('Mlle', 'Miss')

test['Title'] = test['Title'].replace('Ms', 'Miss')

test['Title'] = test['Title'].replace('Mme', 'Mrs')

train['Title_clean'] = train['Title'].astype('category').cat.codes

test['Title_clean'] = test['Title'].astype('category').cat.codes

#age

train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)

test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True)

train.loc[ train['Age'] <= 10, 'Age_clean'] = 0

train.loc[(train['Age'] > 10) & (train['Age'] <= 16), 'Age_clean'] = 1

train.loc[(train['Age'] > 16) & (train['Age'] <= 20), 'Age_clean'] = 2

train.loc[(train['Age'] > 20) & (train['Age'] <= 26), 'Age_clean'] = 3

train.loc[(train['Age'] > 26) & (train['Age'] <= 30), 'Age_clean'] = 4

train.loc[(train['Age'] > 30) & (train['Age'] <= 36), 'Age_clean'] = 5

train.loc[(train['Age'] > 36) & (train['Age'] <= 40), 'Age_clean'] = 6

train.loc[(train['Age'] > 40) & (train['Age'] <= 46), 'Age_clean'] = 7

train.loc[(train['Age'] > 46) & (train['Age'] <= 50), 'Age_clean'] = 8

train.loc[(train['Age'] > 50) & (train['Age'] <= 60), 'Age_clean'] = 9

train.loc[ train['Age'] > 60, 'Age_clean'] = 10

test.loc[ test['Age'] <= 10, 'Age_clean'] = 0

test.loc[(test['Age'] > 10) & (test['Age'] <= 16), 'Age_clean'] = 1

test.loc[(test['Age'] > 16) & (test['Age'] <= 20), 'Age_clean'] = 2

test.loc[(test['Age'] > 20) & (test['Age'] <= 26), 'Age_clean'] = 3

test.loc[(test['Age'] > 26) & (test['Age'] <= 30), 'Age_clean'] = 4

test.loc[(test['Age'] > 30) & (test['Age'] <= 36), 'Age_clean'] = 5

test.loc[(test['Age'] > 36) & (test['Age'] <= 40), 'Age_clean'] = 6

test.loc[(test['Age'] > 40) & (test['Age'] <= 46), 'Age_clean'] = 7

test.loc[(test['Age'] > 46) & (test['Age'] <= 50), 'Age_clean'] = 8

test.loc[(test['Age'] > 50) & (test['Age'] <= 60), 'Age_clean'] = 9

test.loc[ test['Age'] > 60, 'Age_clean'] = 10

#cabin

mapping = {'A': 0,'B': 1,'C': 2,'D': 3,'E': 4,'F': 5,'G': 6,'T': 7}

train['Cabin_clean'] = train['Cabin'].str[:1]

train['Cabin_clean'] = train['Cabin_clean'].map(mapping)

train['Cabin_clean'] = train.groupby('Pclass')['Cabin_clean'].transform('median')

test['Cabin_clean'] = test['Cabin'].str[:1]

test['Cabin_clean'] = test['Cabin_clean'].map(mapping)

test['Cabin_clean'] = test.groupby('Pclass')['Cabin_clean'].transform('median')



feature = ['Pclass','SibSp','Parch','Sex_clean','Embarked_clean','Family','Solo','Title_clean','Age_clean','Cabin_clean','Fare_clean']

label = ['Survived']

x_train = train[feature]

x_test = test[feature]

y_train = train[label]



clf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=11)

clf.fit(x_train, y_train)

gender_data['Survived'] = clf.predict(x_test)

gender_data.to_csv('My_submission.csv',index=False)