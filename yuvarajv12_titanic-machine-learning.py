# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
from xgboost import XGBClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


train_df = pd.read_csv("/kaggle/input/titanic/train.csv")
test_df = pd.read_csv("/kaggle/input/titanic/test.csv")
# The Name,Embarked, Cabin and Ticket column makes no relation with Survived outcome.
# Hence deleting both the columns 
train_df = train_df.drop(columns = ['Name','Ticket','Embarked','Cabin'])
test_df = test_df.drop(columns = ['Name','Ticket','Embarked','Cabin'])

# Replace gender with binary attribute
gender = {'male': 1,'female': 0}
train_df.Sex = [gender[item] for item in train_df.Sex] 
test_df.Sex = [gender[item] for item in test_df.Sex] 
# One-Hot encoding for Pclass column
pclass_dummies = pd.get_dummies(train_df['Pclass'],prefix='Pclass', drop_first=True)

train_df = train_df.drop(columns='Pclass')
train_df = pd.concat([train_df,pclass_dummies],axis=1, sort=False)

pclass_dummies = pd.get_dummies(test_df['Pclass'],prefix='Pclass', drop_first=True)

test_df = test_df.drop(columns='Pclass')
test_df = pd.concat([test_df,pclass_dummies],axis=1, sort=False)
# Replace NA with 0 in Age column
train_df["Age"] = train_df["Age"].fillna(0)
test_df["Age"] = test_df["Age"].fillna(0)
model = XGBClassifier()
y = train_df["Survived"]
X = train_df.drop(columns="Survived")

model.fit(X,y)
model.score(X,y)
prediction = model.predict(test_df)
output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': prediction})
output.to_csv('my_submission.csv', index=False)