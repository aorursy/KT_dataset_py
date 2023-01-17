import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
name = train['Name'].str.split('.', expand=True)

name.columns = ['Family', 'Mr', 't']

name2 = name['Family'].str.split(',', expand=True)



name_test = test['Name'].str.split('.', expand=True)

name_test.columns = ['Family', 'Mr']

name2t = name_test['Family'].str.split(',', expand=True)

name2.columns = ['A1', 'A2']

train['status'] = name2['A2']

name2t.columns = ['A1', 'A2']

test['status'] = name2t['A2']
# SibSp — это число братьев, сестер или супругов на борту у человека

train['Family'] = train['SibSp'] + train['Parch']

test['Family'] = test['SibSp'] + test['SibSp']
train = train.drop(['Name', 'SibSp', 'Parch'], axis=1)

test = test.drop(['Name', 'SibSp', 'Parch'], axis=1)
train['Is_alone'] = train.Family == 0

test['Is_alone'] = test.Family == 0
train['Fare_Category'] = pd.cut(train['Fare'], bins=[0,7.90,14.45,31.28,120], labels=['Low','Mid', 'High_Mid','High'])
test['Fare_Category'] = pd.cut(test['Fare'], bins=[0,7.90,14.45,31.28,120], labels=['Low','Mid', 'High_Mid','High'])
train = train.drop(['Ticket'], axis=1)

test = test.drop(['Ticket'], axis=1)
nulls = train.isna().sum()

nulls[nulls > 0]
test_nulls = test.isna().sum()

test_nulls[test_nulls > 0] 
train['Age'] = train['Age'].fillna(train['Age'].mean())

test['Age'] = test['Age'].fillna(test['Age'].mean())

train['Cabin'] = train['Cabin'].fillna('G6')

test['Cabin'] = test['Cabin'].fillna('B57 B59 B63 B66')

train['Embarked'] = train['Embarked'].fillna('S')

train['Fare_Category'] = train['Fare_Category'].fillna('Mid')

test['Fare_Category'] = test['Fare_Category'].fillna('Low')

test['Fare'] = test['Fare'].fillna(test['Fare'].mean())
y = train.Survived

X = train.drop(['Survived'], axis=1)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
import catboost

from catboost import *
X[:5]
cat_features = ['Pclass', 'Sex', 'Cabin', 'Embarked', 'status', 'Fare_Category']

cat_features
from catboost import CatBoostClassifier

model = CatBoostClassifier(

    iterations = 100,

    learning_rate = 0.1,

    loss_function='CrossEntropy'

)

model.fit(

    X_train, y_train,

    cat_features=cat_features,

    eval_set=(X_test, y_test),

    

)

print('Model is fitted: ' + str(model.is_fitted()))

print('Model params:')

print(model.get_params())
predicted = model.predict(data=X_test)
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, predicted))
preds = model.predict(test)
preds = pd.Series(preds)
gender = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
gender['survived'] = preds
gender = gender.drop(['Survived'], axis=1)

gender.columns = ['PassengerId', 'Survived']
gender['Survived'] = gender['Survived'].astype('int64')
gender.to_csv('submission.csv', index=False)