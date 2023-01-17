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
import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
print(gender_submission.head())

print(test.head())
train.head()
train.drop('PassengerId',axis=1).groupby(['Sex','Pclass','Embarked']).mean()
train.pivot_table('Survived', index='Sex', columns='Pclass', margins=True)
data = train.drop(['Ticket', 'Cabin','Name'], axis=1)

data.head()
data['Embarked'] = data['Embarked'].astype('S')
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

le.fit(data['Sex'])

data['Sex'] = le.transform(data['Sex'])

le.fit(data['Embarked'].astype('S'))

data['Embarked'] = le.transform(data['Embarked'])

data.head()
data.head()
print(data.shape)

print(data.dropna().shape)

data = data.dropna()
Xtrain = data.drop(['Survived','PassengerId'],axis=1)

ytrain = data['Survived']
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=0)

model.fit(Xtrain,ytrain)
test.head()
data_test = test.drop(['PassengerId','Name','Cabin','Ticket'],axis=1)
data_test['Embarked'] = data_test['Embarked'].astype('S')

le = LabelEncoder()

le.fit(data_test['Sex'])

data_test['Sex'] = le.transform(data_test['Sex'])

le.fit(data_test['Embarked'])

data_test['Embarked'] = le.transform(data_test['Embarked'])

data_test.head()
print(any(data_test['Pclass'].isnull()))

print(any(data_test['Sex'].isnull()))

print(any(data_test['Age'].isnull()))

print(any(data_test['SibSp'].isnull()))

print(any(data_test['Parch'].isnull()))

print(any(data_test['Fare'].isnull()))

print(any(data_test['Embarked'].isnull()))
data_test['Age'][data_test['Age'].isnull()] = data_test['Age'].mean()

data_test['Fare'][data_test['Fare'].isnull()] = data_test['Fare'].mean()
print(any(data_test['Pclass'].isnull()))

print(any(data_test['Sex'].isnull()))

print(any(data_test['Age'].isnull()))

print(any(data_test['SibSp'].isnull()))

print(any(data_test['Parch'].isnull()))

print(any(data_test['Fare'].isnull()))

print(any(data_test['Embarked'].isnull()))
Xtest = data_test

ytest = gender_submission['Survived']
ypred = model.predict(Xtest)

from sklearn import metrics

print(metrics.classification_report(ypred, ytest))
import seaborn as sns

import matplotlib.pyplot as plt

sns.set() # настройка стиля

from sklearn.metrics import confusion_matrix

mat = confusion_matrix(ytest, ypred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
metrics.accuracy_score(ypred, ytest)
result = pd.DataFrame({'test': ytest,'pred': ypred})
result.to_csv('result_titanic')