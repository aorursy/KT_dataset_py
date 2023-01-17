import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_path = '../input/home-data-for-ml-course/train.csv'

test_path = '../input/home-data-for-ml-course/test.csv'



train_data = pd.read_csv(train_path)

test_data = pd.read_csv(test_path)
train_data.head()
train_features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = train_data[train_features]



y = train_data.SalePrice
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=42)
# Decision Tree

FirstModel = DecisionTreeRegressor(random_state=1)



# Fit Model

FirstModel.fit(train_X, train_y)



# Predict

pred_y = FirstModel.predict(test_X)



# Checking Error

error_mae = mean_absolute_error(pred_y, test_y)



# Print Error

print('Error is: {:,.4f}'.format(error_mae))
# Decision Tree with 100 Max Nodes

SecondModel = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)



# Fit Model

SecondModel.fit(train_X, train_y)



# Predict

pred_y = SecondModel.predict(test_X)



# Checking Error

error_mae = mean_absolute_error(pred_y, test_y)



# Print Error

print('Error is: {:,.4f}'.format(error_mae))
# Random Forest Tree

ThirdModel = RandomForestRegressor(random_state=1)



# Fit Model

ThirdModel.fit(train_X, train_y)



# Predict

pred_y = ThirdModel.predict(test_X)



# Checking Error

error_mae = mean_absolute_error(pred_y, test_y)



# Print Error

print('Error is: {:,.4f}'.format(error_mae))
# Now on Test Data

X_test = test_data[train_features]



pred_test = ThirdModel.predict(X_test)



output = pd.DataFrame({

    'Id': test_data.Id,

    'SalePrice': pred_test

})



output.to_csv('CompetitionSubmission.csv')
output.head()
output.shape