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
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
# トレーニングデータの欠損値Ageを補う

train_data['AgeBand'] = pd.cut(train_data['Age'], 5)

train_data[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
train_data.loc[ train_data['Age'] <= 16, 'Age'] = 0

train_data.loc[ (train_data['Age'] > 16) & (train_data['Age'] <= 32), 'Age'] = 1

train_data.loc[ (train_data['Age'] > 32) & (train_data['Age'] <= 48), 'Age'] = 2

train_data.loc[ (train_data['Age'] > 48) & (train_data['Age'] <= 64), 'Age'] = 3

train_data.loc[ train_data['Age'] > 64, 'Age'] = 4
test_data.loc[ test_data['Age'] <= 16, 'Age'] = 0

test_data.loc[ (test_data['Age'] > 16) & (test_data['Age'] <= 32), 'Age'] = 1

test_data.loc[ (test_data['Age'] > 32) & (test_data['Age'] <= 48), 'Age'] = 2

test_data.loc[ (test_data['Age'] > 48) & (test_data['Age'] <= 64), 'Age'] = 3

test_data.loc[ test_data['Age'] > 64, 'Age'] = 4
train_data.isnull().sum()
train_data.groupby('Age').size()
test_data.groupby('Age').size()
# 一番ウェイトの大きい年齢帯1（16 - 32歳）で欠損値を埋める

train_data['Age'] = train_data['Age'].fillna(1)

train_data.groupby('Age').size()
# 一番ウェイトの大きい年齢帯1（16 - 32歳）で欠損値を埋める

test_data['Age'] = test_data['Age'].fillna(1)

test_data.groupby('Age').size()
from sklearn.ensemble import RandomForestClassifier



y = train_data['Survived']



features = ['Pclass', "Sex", "SibSp", "Parch", "Age"]

X = pd.get_dummies(train_data[features])

#print(train_data[features])

#print(X.head())

X_test = pd.get_dummies(test_data[features])



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId' : test_data.PassengerId, 'Survived' : predictions})

output.to_csv('my_submission.csv', index=False)

print('Your submission was successfully saved!')