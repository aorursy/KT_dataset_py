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
feature = ["budget","popularity","runtime"]

label = "revenue"
X_train = train[feature]
Y = train[label]
X_train.isnull().sum()
X_train["runtime"] = X_train["runtime"].fillna(0)
X_train.isnull().sum()
Y.isnull().sum()
MLPA = MLPClassifier(hidden_layer_sizes=(50,),max_iter=200)
MLPA.fit(X_train,Y)
MLPA.score(X_train,Y)
X_test = test[feature]
X_test["runtime"] = X_test["runtime"].fillna(0)
X_test.isnull().sum()
MLPA.predict(X_test)
X_test["revenue"] = MLPA.predict(X_test)