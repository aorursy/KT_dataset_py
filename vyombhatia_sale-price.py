import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Importing Libraries for plotting purposes:

import matplotlib.pyplot as plt

import seaborn as sns



# Importing Libraries for preprocessing and feature engineering:

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

from category_encoders import CatBoostEncoder

from sklearn.impute import SimpleImputer



# Importing libraries for training and predecting of the data:

from sklearn.ensemble import RandomForestRegressor 

from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRegressor



# Importing Metrics to measure model accuracy:

from sklearn.metrics import r2_score, mean_squared_error
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train.head()
y = train['SalePrice']

ID = test['Id']

train.drop(['Id', 'SalePrice'], axis=1, inplace=True)

test.drop(['Id'], axis=1, inplace=True)
nullcol  = pd.DataFrame(list(train.isnull().sum()), columns=["examplesnull"])

cols = pd.DataFrame(train.columns, columns=['column name'])

nullinfo = pd.concat([cols, nullcol], axis=1)

s = (train.dtypes == 'object')

catcol = list(s[s].index)
catsimp = SimpleImputer(strategy='most_frequent')
cattrain = pd.DataFrame(catsimp.fit_transform(train[catcol]), columns=catcol)

cattest = pd.DataFrame(catsimp.fit_transform(test[catcol]), columns=catcol)
n = (train.dtypes == ('int64', 'float64'))

numcol = list(n[n].index)
numsimp = SimpleImputer(strategy='median')
numtrain = pd.DataFrame(numsimp.fit_transform(train[numcol]), columns=numcol)

numtest = pd.DataFrame(numsimp.fit_transform(test[numcol]), columns=numcol)

enctrain = cattrain.copy()



LEnc = LabelEncoder()



for i in catcol:

    enctrain[i] = LEnc.fit_transform(enctrain[i])
LEnc = LabelEncoder()



enctest = cattest.copy()

for i in catcol:

    enctest[i] = LEnc.fit_transform(enctest[i])
newtrain = pd.concat([enctrain, numtrain], axis=1)

newtest = pd.concat([enctest, numtest], axis=1)
scale = MinMaxScaler()



scaledtrain = pd.DataFrame(newtrain, columns=newtrain.columns)

scaledtest = pd.DataFrame(newtest, columns=newtrain.columns)
from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(scaledtrain, y, train_size=0.7, test_size=0.3)


RanModel = RandomForestRegressor(n_estimators=500)

RanModel.fit(xtrain, ytrain)



RanPreds = RanModel.predict(xtest)

    

print("The R2 Score of this model is:", r2_score(RanPreds, ytest))
TreeModel = DecisionTreeRegressor()

TreeModel.fit(xtrain, ytrain)



TreePreds = TreeModel.predict(xtest)

print("The R2 Score of this model is:", r2_score(TreePreds, ytest))
XGmodel = XGBRegressor(n_estimators=500)

XGmodel.fit(xtrain, ytrain)



XGpreds = XGmodel.predict(xtest)

print("The R2 Score of this model is:", r2_score(ytest, XGpreds))


sns.scatterplot(x=xtest['TotalBsmtSF'], y=ytest)

sns.scatterplot(x=xtest['TotalBsmtSF'], y=XGpreds)
predictions = RanModel.predict(scaledtest)
submit= pd.DataFrame()

submit['id'] = ID

submit['SalePrice'] = predictions



submit.to_csv('submission.csv', index = False)

submit.head()