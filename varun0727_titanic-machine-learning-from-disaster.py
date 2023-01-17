# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
train_data.info()
train_data.head(5)
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
sns.heatmap(pd.isnull(train_data), cbar=None, cmap='Blues', xticklabels=True, yticklabels=False)
train_data['Age'].fillna(np.floor(train_data['Age'].mean()), inplace=True)

test_data['Age'].fillna(np.floor(test_data['Age'].mean()), inplace=True)

test_data['Fare'].fillna(np.floor(test_data['Fare'].mean()), inplace=True)
train_data.dropna(subset=['Embarked'], inplace=True)

train_data.reset_index(drop=True, inplace=True)
label = train_data['Survived']

PassengerId = test_data['PassengerId']

train_data.drop(['Name', 'PassengerId', 'Survived', 'Ticket', 'Cabin'], axis=1, inplace=True)

test_data.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=True)

train_data.head()
train_data[:5]
from sklearn.preprocessing import OneHotEncoder,  MaxAbsScaler
encoder = OneHotEncoder()

temp1 = encoder.fit_transform(train_data[['Sex', 'Embarked']])

temp2 = encoder.fit_transform(test_data[['Sex', 'Embarked']])

temp1 = pd.DataFrame(temp1.toarray(), columns=['Male', 'Female', 'S', 'C', 'Q'])

temp2 = pd.DataFrame(temp2.toarray(), columns=['Male', 'Female', 'S', 'C', 'Q'])

temp1.head()
train_data.drop(['Sex', 'Embarked'], axis=1, inplace=True)

train_data = pd.concat([train_data, temp1], axis=1)

test_data.drop(['Sex', 'Embarked'], axis=1, inplace=True)

test_data = pd.concat([test_data, temp2], axis=1)

train_data.head()
maxabs = MaxAbsScaler()

temp_1 = maxabs.fit_transform(train_data[['Age', 'Fare']])

train_data.drop(['Age', 'Fare'], axis=1, inplace=True)

train_data = pd.concat([pd.DataFrame(data = temp_1, columns=['Age', 'Fare']), train_data], axis=1)

temp_2 = maxabs.fit_transform(test_data[['Age', 'Fare']])

test_data.drop(['Age', 'Fare'], axis=1, inplace=True)

test_data = pd.concat([pd.DataFrame(data = temp_2, columns=['Age', 'Fare']), test_data], axis=1)

train_data.head()
test_data.head()
test_data.info()
from sklearn.svm import SVC
model = SVC()

model.fit(train_data, label)

result = model.predict(test_data)
Submission = pd.concat([PassengerId, pd.DataFrame(result, columns=['Survived'])], axis=1)
Submission.to_csv('result.csv', index=False)