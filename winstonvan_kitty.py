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
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
train_data_categorical = train_data.select_dtypes(exclude='number')
train_data_categorical_1he = pd.get_dummies(train_data_categorical)
train_data_numerical = train_data.select_dtypes(include=[np.number])
train_data = pd.concat([train_data_categorical_1he, train_data_numerical], axis='columns')
train_data.fillna(0, inplace=True)
y = train_data.SalePrice
X = train_data.select_dtypes(include=[np.number]).drop(['SalePrice'], axis=1) # Can also include Id
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler(copy=False)
# X = scaler.fit_transform(X, y)
X.shape
# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer()
# data_with_imputed_values = imputer.fit_transform(X, y)
test_data_categorical = test_data.select_dtypes(exclude='number')
test_data_categorical = pd.get_dummies(test_data_categorical)
test_data_numerical = test_data.select_dtypes(include=[np.number])
test_data = pd.concat([test_data_categorical, test_data_numerical], axis='columns')
test_data.fillna(0, inplace=True)
print(test_data_categorical.shape)
print(test_data_numerical.shape)
print(test_data.shape)
# Missing features between train and test
missing_cols = set(train_data.columns) - set(test_data.columns)
# Default 0
for c in missing_cols:
    test_data[c] = 0
# Set order of the columns is the same
test_data = test_data[train_data.columns]
test_data = test_data.drop(['SalePrice'], axis='columns')
from xgboost import XGBRegressor

model = XGBRegressor(n_estimators=5000)
model.fit(X, y, early_stopping_rounds=5, eval_set=[(X, y)], verbose=False)

predictions = model.predict(test_data)
submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': predictions})
submission.to_csv('submission.csv', index=False)

