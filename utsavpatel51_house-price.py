import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

import seaborn

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
print(train['SalePrice'].isnull().sum())

y = train['SalePrice']

X = train.drop(['SalePrice'],axis=1)
missing_col = [col for col in X.columns if X[col].isnull().any()]

per = [[col,X[col].isnull().sum()/X[col].isnull().count()*100] for col in missing_col]

per = sorted(per,key = lambda x:x[1],reverse=True)

per
drop_col = [val[0] for val in per[:6]]

X = X.drop(drop_col,axis=1)

X_test = test.drop(drop_col,axis=1)
print(X.shape)

print(X_test.shape)
object_cols = [col for col in X.columns if X[col].dtype == "object"]

other_cols  = [col for col in X.columns if X[col].dtype != "object"]
low_cardinality_cols = [col for col in object_cols if X[col].nunique() < 10]





high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))



print('Categorical columns that will be one-hot encoded:', low_cardinality_cols)

print('\nCategorical columns that will be dropped from the dataset:', high_cardinality_cols)
X = X.fillna(method='ffill')

X_test = X_test.fillna(method='ffill')
from sklearn.preprocessing import OneHotEncoder



ohe = OneHotEncoder(handle_unknown='ignore',sparse=False)

OH_X = pd.DataFrame(ohe.fit_transform(X[low_cardinality_cols]))

OH_X_test = pd.DataFrame(ohe.transform(X_test[low_cardinality_cols]))



OH_X.index = X.index

OH_X_test.index = X_test.index



OH_X = pd.concat([OH_X,X.drop(object_cols,axis=1)],axis=1)

OH_X_test = pd.concat([OH_X_test,X_test.drop(object_cols, axis=1)], axis=1)
model = RandomForestRegressor(n_estimators=100, random_state=0)

model.fit(OH_X, y)

preds = model.predict(OH_X_test)
output = pd.DataFrame({'Id': test.Id,

                       'SalePrice': preds})

output.to_csv('submission.csv', index=False)