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
data_train = pd.read_csv("/kaggle/input/titanic/train.csv")

print (data_train.shape)

print (data_test.shape)
feature_set = data_train[['Pclass', 'Sex','Age','SibSp','Parch','Fare','Cabin']]

target = data_train[['Survived']]
target.Survived.unique()
target['Survived'].value_counts()
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(feature_set, target, test_size=0.2, random_state=42)



print(x_train.head())

print(y_train.head())
x_train.Cabin.unique()
x_train.Fare.unique()
x_train = x_train.drop(['Cabin'], axis=1)

x_test = x_test.drop(['Cabin'], axis=1)

x_train.head()
from sklearn.preprocessing import LabelEncoder

x_train['Sex'] = LabelEncoder().fit_transform(x_train['Sex'])

x_test['Sex'] = LabelEncoder().fit_transform(x_test['Sex'])

print(x_train.head())
x_train.describe()
x_train.fillna(x_train.mean(), inplace=True)

x_test.fillna(x_train.mean(), inplace=True)
x_train.describe()
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)
data_test =  pd.read_csv("/kaggle/input/titanic/test.csv")
feature_set_test = data_test[['Pclass', 'Sex','Age','SibSp','Parch','Fare']]

feature_set_test['Sex'] = LabelEncoder().fit_transform(feature_set_test['Sex'])

feature_set_test.fillna(feature_set_test.mean(), inplace=True)

feature_set_test.head()
feature_set_test.describe()
pred_list = lr.predict(feature_set_test)

y_pred = pd.DataFrame(pred_list,columns=['Survived'])

predictions = pd.concat([data_test['PassengerId'],y_pred], axis=1)

predictions.head()