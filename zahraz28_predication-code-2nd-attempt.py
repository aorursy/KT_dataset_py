from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
import numpy as np 
import pandas as pd 
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

import os
print(os.listdir("../input"))

train_data = pd.read_csv('../input/train.csv')
y = train_data.SalePrice
predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']
X = train_data[predictor_cols]

model_train_X, model_test_X, model_train_y, model_test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25)
my_imputer = Imputer()
model_train_X = my_imputer.fit_transform(model_train_X)
model_test_X = my_imputer.transform(model_test_X)

my_model = XGBRegressor()
my_model.fit(X,y, verbose=False)

test = pd.read_csv('../input/test.csv')

test_X = test[predictor_cols]

predicted_prices = my_model.predict(test_X)


my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})

my_submission.to_csv('submission.csv', index=False)












