# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC  
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/diabetes.csv")
df.info()
df.head()
df.hist(figsize=(10,8))
X = df[['DiabetesPedigreeFunction']]
y = df['Outcome']
model = LinearSVC()
model.fit(X,y)
predicted = model.predict(X)
accuracy_score(y,predicted)
model = KNeighborsClassifier()
model.fit(X,y)
predicted = model.predict(X)
print(accuracy_score(y,predicted))
model = NearestCentroid()
model.fit(X, y)
predicted = model.predict(X)
accuracy_score(y,predicted)
model = svm.SVC()
model.fit(df[['DiabetesPedigreeFunction']],df['Outcome'])
pridected = model.predict(df[['DiabetesPedigreeFunction']])
print(accuracy_score(df['Outcome'],predicted))
model = RandomForestClassifier()
model.fit(X,y)
predicted = model.predict(X)
accuracy_score(y,predicted)
model = DecisionTreeClassifier()
model.fit(X, y)
predicted = model.predict(X)
accuracy_score(y,predicted)
model = ExtraTreesClassifier()
model.fit(X, y)
predicted = model.predict(X)
accuracy_score(y,predicted)
model = AdaBoostClassifier()
model.fit(X, y)
predicted = model.predict(X)
accuracy_score(y,predicted)
model = GradientBoostingClassifier()
model.fit(X, y)
predicted = model.predict(X)
accuracy_score(y,predicted)
from sklearn.model_selection import train_test_split
outcome=df['Outcome']
data=df[df.columns[:8]]
train,test=train_test_split(df,test_size=0.25,random_state=0,stratify=df['Outcome'])# stratify the outcome
train_X=train[train.columns[:8]]
test_X=test[test.columns[:8]]
train_Y=train['Outcome']
test_Y=test['Outcome']
model = ExtraTreesClassifier(n_estimators=10)
model.fit(train_X, train_Y)
predicted = model.predict(test_X)
accuracy_score(test_Y,predicted)
model = DecisionTreeClassifier()
model.fit(train_X,train_Y)
predicted = model.predict(test_X)
accuracy_score(test_Y,predicted)
