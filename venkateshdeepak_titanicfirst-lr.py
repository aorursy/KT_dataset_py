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
import matplotlib.pyplot as plt

import pandas as pd

from sklearn import preprocessing

from sklearn import linear_model 
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.isnull().sum()
train.head()
test.head()
feature = ["Sex", "Pclass","SibSp","Parch"]

labels = ["Survived"]
X = train[feature].copy()

Y = train[labels].values
enc = preprocessing.OrdinalEncoder()

enc.fit(X["Sex"].values.reshape(-1,1))
enc.categories_
X["Sex"] = enc.transform(X["Sex"].values.reshape(-1,1))
model = linear_model.LogisticRegression()
X.shape
Y.shape
type(X),type(Y)
model.fit(X.values,Y)

model.score(X.values,Y)
enc.categories_
model.predict([[1, 1,1,1],])
X_test = test[feature].copy()
X_test.isnull().sum()
X_test["Sex"] = enc.transform(X_test["Sex"].values.reshape(-1,1))
X_test = X_test.fillna(0)
test["Survived"] = model.predict(X_test.values)
test[["PassengerId","Survived"]].to_csv("submit.csv", index=False)