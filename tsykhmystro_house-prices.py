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



data = pd.read_csv('../input/train.csv')

data.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = data.SalePrice

X = data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])

train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25)



my_imputer = Imputer()

train_X = my_imputer.fit_transform(train_X)

test_X = my_imputer.transform(test_X)
from xgboost import XGBRegressor



my_model = XGBRegressor()

# Add silent=True to avoid printing out updates with each cycle

my_model.fit(train_X, train_y, verbose=False)
# make predictions

predictions = my_model.predict(test_X)



from sklearn.metrics import mean_absolute_error

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
# Read the test data

test = pd.read_csv('../input/test.csv')



# Treat the test data in the same way as training data. In this case, pull same columns.

test_X = test.select_dtypes(exclude=['object'])

test_X = my_imputer.transform(test_X)



# Use the model to make predictions

predicted_prices = my_model.predict(test_X)

# We will look at the predicted prices to ensure we have something sensible.

print(predicted_prices)
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)