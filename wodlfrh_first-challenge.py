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
train_raw = pd.read_csv('../input/train.csv')
train_raw.head()
test_raw = pd.read_csv('../input/test.csv')
train_raw.describe()
test_raw.describe()
train_raw.columns
y = train_raw.Survived
select_feature = ['Pclass','Sex','Age','SibSp','Parch','Fare']
X = train_raw[select_feature]
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(X.Sex)
X.Sex = le.transform(X.Sex)
X.describe()
from sklearn.ensemble import RandomForestRegressor
first_model = RandomForestRegressor(random_state=0)
X.describe()
y.describe()
X.isnull().sum()
X.Age.fillna(X.Age.mean(),inplace=True)
X.Age.isnull().sum
first_model.fit(X,y)
test_X = test_raw[select_feature]
test_X.describe()
test_X.fillna(test_X.Age.mean(),inplace=True)
le2 = LabelEncoder()
le2.fit(test_X.Sex)
test_X.Sex = le2.transform(test_X.Sex)
first_model.predict(test_X)
from sklearn.tree import DecisionTreeClassifier
sec_model = DecisionTreeClassifier()
sec_model.fit(X,y)
sec_model.predict(test_X)
result = sec_model.predict(test_X)
P_ID = np.array(test_raw['PassengerId'])
P_ID
solution = pd.DataFrame(result,P_ID,columns=['Survived'])
solution
solution.to_csv("jay_first.csv", index_label='PassengerId')