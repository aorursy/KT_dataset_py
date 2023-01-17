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
from sklearn.ensemble import RandomForestRegressor
#read
train = pd.read_csv('../input/train.csv')
#y
train_y = train.SalePrice
predictor_cols =  ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd','TotalBsmtSF','BsmtUnfSF','FullBath','HalfBath']

#train x 
train_X = train[predictor_cols]
model = RandomForestRegressor()
model.fit(train_X,train_y)
#read test data
from sklearn.preprocessing import Imputer
test = pd.read_csv('../input/test.csv')
test_X = test[predictor_cols]
my_imputer = Imputer()

#one_hot_encoded_encoding
one_hot_encoded_training_predictors = pd.get_dummies(train_X)
one_hot_encoded_test_predictors = pd.get_dummies(test_X)

final_train, final_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,
                                                                    join='left', 
                                                                    axis=1)
#imputation to handle missing values
data_with_imputed_values = my_imputer.fit_transform(final_test)

predicted_prices = model.predict(data_with_imputed_values)
# setting up submission
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
my_submission.to_csv('submission.csv', index=False)
