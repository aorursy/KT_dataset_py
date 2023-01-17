# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn import preprocessing

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head()
train.Street.value_counts()
train.MSSubClass.value_counts()
train.groupby(train.MSSubClass)["SalePrice"].agg({"SalePriceAVG":np.mean}).sort_values("SalePriceAVG")
train["KitchenQual"].unique()
train["KitchenQual"].value_counts()
train["KitchenQual"].isnull().sum()
feature = ["LotArea","GarageArea","BedroomAbvGr","GrLivArea","KitchenAbvGr","YrSold","FullBath","HalfBath","YearBuilt","TotRmsAbvGrd","MoSold"]

label = ["SalePrice"]
X_train = train[feature].copy()

X_train.isnull().sum()
train.columns
ore = preprocessing.OneHotEncoder()
ore.fit(train["KitchenQual"].values.reshape(-1,1))
ore.transform(train["KitchenQual"].values.reshape(-1,1)).toarray()
X_train["kq1"] = 1

X_train["kq2"] = 1

X_train["kq3"] = 1

X_train["kq4"] = 1
X_train[["kq1","kq2","kq3","kq4"]] = ore.transform(train["KitchenQual"].values.reshape(-1,1)).toarray()
ore.categories_
lm = LinearRegression()
lm.fit(X_train,train[label])
lm.coef_,lm.intercept_
lm.score(X_train,train[label])
test[feature].isnull().sum()
test["KitchenQual"].isnull().sum()

test["KitchenQual"].unique()
test.fillna(0,inplace = True)

test.loc[test["KitchenQual"]==0,"KitchenQual"] = "TA"

x_test = test[feature].copy()

x_test = x_test.fillna(0)

x_test["kq1"] = 1

x_test["kq2"] = 1

x_test["kq3"] = 1

x_test["kq4"] = 1

x_test[["kq1","kq2","kq3","kq4"]] = ore.transform(test["KitchenQual"].values.reshape(-1,1)).toarray()
x_test.columns
lm.coef_.shape
lm.intercept_
len(feature)
x_test.values.shape
test["SalePrice"] = lm.predict(x_test.values)
lm.predict(x_test.values)
X_train
x_test.shape,X_train.shape
test[["Id","SalePrice"]].to_csv("submit.csv",index= False)
import pickle
pickle.dump(lm, open('housing_lm_v1.pkl', 'wb'))