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
df = pd.read_csv('../input/train.csv')
df = pd.get_dummies(df,columns = ['Sex'])
cols_to_drop = ['PassengerId','Name','Ticket','Cabin','Embarked']
df_new = df.drop(cols_to_drop,axis=1)
X_train = df_new.drop(['Survived'],axis=1)
X_train['Age'] = X_train['Age'].fillna(X_train['Age'].mean())
y_train = df_new['Survived']
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train,y_train)
from sklearn.metrics import accuracy_score 
df_test = pd.read_csv('../input/test.csv')
df_test_new = df_test.drop(cols_to_drop,axis=1)
df_test_new = pd.get_dummies(df_test_new,columns = ['Sex'])
X_test = df_test_new
X_test['Age'] = X_test['Age'].fillna(X_test['Age'].mean())
X_test['Fare'] = X_test['Fare'].fillna(X_test['Fare'].mean())
y_test = clf.predict(X_test)
df_test['prediction'] = y_test
df_test[['PassengerId','prediction']]
