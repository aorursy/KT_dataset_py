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
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

from sklearn.model_selection import train_test_split

sns.set(style="white", color_codes=True)



data_raw = pd.read_csv('../input/train.csv')

X_test = pd.read_csv('../input/test.csv')

data_raw['FamilySize'] = data_raw['SibSp']+data_raw['Parch']

data_raw.Age[data_raw['Age'].isna()] = data_raw['Age'].mean()

data = data_raw.drop(['Embarked','Cabin','Fare','Ticket','Name','SibSp','Parch'],axis=1)

data = pd.get_dummies(data,prefix=['Sex'])

X_train, y_train = data[data.columns.drop('Survived')], data['Survived']



from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=90, max_depth=5, random_state=0,verbose=True)

clf.fit(X_train[X_train.columns[1:]],y_train)  

clf.score(X_train[X_train.columns[1:]],y_train)

X_test['FamilySize'] = X_test['SibSp']+ X_test['Parch']

X_test.Age[X_test['Age'].isna()] = X_test['Age'].mean()

data2 = X_test.drop(['Embarked','Cabin','Fare','Ticket','Name','SibSp','Parch'],axis=1)

data2 = pd.get_dummies(data2,prefix=['Sex'])



y_hat = clf.predict(data2[data2.columns[1:]])

df = pd.DataFrame({'PassengerId': data2.PassengerId, 'Survived':y_hat })

df.to_csv('submission.csv', index=False)