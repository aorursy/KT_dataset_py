# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()
train.columns
feature = ["SibSp","Age","Pclass","Fare"]
label = "Survived"
X_train = train[feature]
Y = train[label]
enc = preprocessing.OrdinalEncoder()
enc.fit(train['Sex'].values.reshape(-1,1))
enc.categories
X_train["Male"] = 0
X_train["Female"] = 1
X_train.isnull().sum()
X_train["Age"] = X_train["Age"].fillna(0)
MLPA = MLPClassifier(hidden_layer_sizes=(10,),max_iter=20)
MLPA.fit(X_train,Y)
MLPA.score(X_train,Y)
X_test = test[feature]
enc.fit(test["Sex"].values.reshape(-1,1))
X_test["Male"] = 0
X_test["Female"] = 1
X_test.isnull().sum()
X_test["Fare"].fillna(0)
X_test.isnull().sum()
X_test["Fare"] = X_test["Fare"].fillna(0)
X_test["Age"] = X_test["Age"].fillna(0)
X_test.isnull().sum()
MLPA.predict(X_test)
test["Survived"] = MLPA.predict(X_test)
test[["PassengerId","Survived"]].to_csv("NEW MLP",index = False)
