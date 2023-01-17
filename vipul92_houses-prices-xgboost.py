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
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer

train = pd.read_csv('../input/train.csv')
y=train.SalePrice
predictor_cols =  ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd','TotalBsmtSF','BsmtUnfSF','FullBath','HalfBath']
X = train[predictor_cols]
train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25)

from xgboost import XGBRegressor
my_imputer = Imputer()
train_X = my_imputer.fit_transform(train_X)
test_X = my_imputer.transform(test_X)

my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], verbose=False)
test = pd.read_csv('../input/test.csv')
test[predictor_cols].describe()
my_imputer = Imputer()
predicted_prices = my_model.predict(my_imputer.fit_transform(test[predictor_cols]))

# setting up submission
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
my_submission.to_csv('submission.csv', index=False)