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
from sklearn import preprocessing

from sklearn import svm
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head()
feature = ["Sex","Pclass","SibSp","Parch"]

labels = ["Survived"]
X = train[feature].copy()

Y = train[labels].values
enc = preprocessing.OrdinalEncoder()

X["Sex"] = enc.fit_transform(X["Sex"].values.reshape(-1,1))
enc.categories_
model = svm.SVC()
model.fit(X.values,Y)
X.isnull().sum()
X_test = test[feature].copy()
X_test["Sex"] = enc.transform(X_test["Sex"].values.reshape(-1,1))
X_test.isnull().sum()
test["Survived"] = model.predict(X_test.values)
test[["PassengerId","Survived"]].to_csv("Submit.csv",index = False)