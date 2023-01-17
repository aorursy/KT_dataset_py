# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/turnover.csv')
df.head()
#make categorical data to integer
df=pd.get_dummies(df)
df.head()
#check if any null values in data by visualize//
print(df.isnull().values.any())
# apply data modelling
y=df.left
X=df.drop('left',axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=99,stratify=y)
model=tree.DecisionTreeClassifier()
model.fit(X_train,y_train)
#predict employees who left
pred=model.predict(X_test)
print(model.score(X_train, y_train)*100)
acc=accuracy_score(y_test,pred)
print(acc)
