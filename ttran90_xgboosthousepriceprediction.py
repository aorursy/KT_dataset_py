# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer

# 1st version take into account only numerical data
# train.dropna(axis=0, subset=['SalePrice'], inplace=True)
# y = train.SalePrice
# X = train.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])
# val_X = test.select_dtypes(exclude=['object'])

# 2nd version apply one-hot-encode to categorical data before imputing
train.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = train.SalePrice
X = train.drop(['Id','SalePrice'], axis=1)
val_X = test.drop(['Id'], axis=1)

one_hot_encoded_train = pd.get_dummies(X)
one_hot_encoded_test = pd.get_dummies(val_X)
X, val_X = one_hot_encoded_train.align(one_hot_encoded_test,
                                                            join='inner', 
                                                            axis=1)

train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.2)

# simple imputer
my_imputer = Imputer()
train_X = my_imputer.fit_transform(train_X)
test_X = my_imputer.transform(test_X)
final_test = my_imputer.transform(val_X)

# Any results you write to the current directory are saved as output.
# try to make a pipeline for clearer code
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor

my_pipeline = make_pipeline(Imputer(), XGBRegressor(n_estimators=1000, learning_rate=0.05))
my_pipeline.fit(train_X, train_y, xgbregressor__early_stopping_rounds = 20, 
        xgbregressor__eval_set= [(test_X, test_y )],
        xgbregressor__verbose= False)

predictions = my_pipeline.predict(test_X)

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
# train the model (without pipeline)
# from xgboost import XGBRegressor
# my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
# my_model.fit(train_X, train_y, early_stopping_rounds=5, 
#              eval_set=[(test_X, test_y)], verbose=False)

# predictions = my_model.predict(test_X)

# from sklearn.metrics import mean_absolute_error
# print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
# create competition predictions
predicted = my_pipeline.predict(final_test)
print(predicted)
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)