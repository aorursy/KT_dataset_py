# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LinearRegression

from sklearn import preprocessing



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train_V2.csv")

test = pd.read_csv("../input/test_V2.csv")
train.info()
train.head()
train.groupby(train.assists)["winPlacePerc"].agg({"winPlacePercAVG":np.mean}).sort_values("winPlacePercAVG")
train["winPlacePerc"].unique()
train["winPlacePerc"].value_counts()
feature = ["damageDealt","killPlace","killPoints","matchDuration","maxPlace","numGroups","rankPoints"]

labels = ["winPlacePerc"]
train[feature].copy()

train[labels].values
lm = LinearRegression()
train[labels].isnull().sum()
ors = preprocessing.OrdinalEncoder()
ors.fit(train["killPlace"].values.reshape(-1,1))
ors.categories_
train[labels].isnull().sum()
lm.fit(train[feature],train[labels].fillna(0))
lm.coef_,lm.intercept_
lm.score(train[feature],train[labels].fillna(0))
X_test = test[feature]
X_test.isnull().sum()
lm.predict(X_test)
test["winPlacePerc"] = lm.predict(X_test)
test[["Id","winPlacePerc"]].to_csv("Submission.csv",index = False)