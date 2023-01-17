import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

example = pd.read_csv('/kaggle/input/titanic/gender_submission.csv') 

test = pd.read_csv('/kaggle/input/titanic/test.csv') 

train = pd.read_csv('/kaggle/input/titanic/train.csv') 

# test.describe()

# train.describe()

# train[train['Fare']> 100].sort_values('Ticket')

# train['Embarked'].value_counts()
train['Embarked'] = train['Embarked'].fillna('S')

train['Age'] = train['Age'].fillna(28.0)

train = train[['Pclass',  'Sex', 'Age', 'SibSp',

       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Survived']] 

train['Ticket'] = train['Ticket'].apply(len)

train['Cabin'] = train['Cabin'].fillna('')

train['Cabin'] = train['Cabin'].apply(len)

train['Sex'] = train['Sex'].replace({'male':1, 'female':2})

train['Embarked'] = train['Embarked'].replace({'S': 1, 'C': 2, 'Q': 3})
test['Embarked'] = test['Embarked'].fillna('S')

test['Age'] = test['Age'].fillna(28.0)

nomera = test['PassengerId']

test = test[['Pclass',  'Sex', 'Age', 'SibSp',

       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']] 

test['Ticket'] = test['Ticket'].apply(len)

test['Cabin'] = test['Cabin'].fillna('')

test['Cabin'] = test['Cabin'].apply(len)

test['Sex'] = test['Sex'].replace({'male':1, 'female':2})

test['Embarked'] = test['Embarked'].replace({'S': 1, 'C': 2, 'Q': 3})
import xgboost as xgb

x = train[train.columns[:-1]]

y = train[train.columns[-1]]

model = xgb.XGBClassifier()

model.fit(x, y)

otvet = model.predict(test)
nomera = pd.DataFrame(nomera)

nomera['Survived']=otvet

nomera.to_csv('otvet.csv',index=False)
