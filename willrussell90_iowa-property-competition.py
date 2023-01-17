# Code you have previously used to load data

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from learntools.core import *







# Path of the file to read. We changed the directory structure to simplify submitting to a competition

iowa_file_path = '../input/train.csv'



home_data = pd.read_csv(iowa_file_path)

# Create target object and call it y

y = home_data.SalePrice

# Create X

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = home_data[features]



# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



# Specify Model

iowa_model = DecisionTreeRegressor(random_state=1)

# Fit Model

iowa_model.fit(train_X, train_y)



# Make validation predictions and calculate mean absolute error

val_predictions = iowa_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))



# Using best value for max_leaf_nodes

iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)

iowa_model.fit(train_X, train_y)

val_predictions = iowa_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))



# Define the model. Set random_state to 1

rf_model = RandomForestRegressor(random_state=1)

rf_model.fit(train_X, train_y)

rf_val_predictions = rf_model.predict(val_X)

rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)



print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

# To improve accuracy, create a new Random Forest model which you will train on all training data

rf_model_on_full_data = RandomForestRegressor()



# fit rf_model_on_full_data on all data from the training data

rf_model_on_full_data.fit(X, y)

# path to file you will use for predictions

test_data_path = '../input/test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)



# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features

test_X = test_data[features]



# make predictions which we will submit. 

test_preds = rf_model_on_full_data.predict(test_X)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



#output = pd.DataFrame({'Id': test_data.Id,

#                       'SalePrice': test_preds})

#print(output)

#output.to_csv('submission.csv', index=False)
home_data.columns.sort_values()

home_data.describe()
updatedFeatures = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 

                   'TotRmsAbvGrd', 'OverallCond', 'TotalBsmtSF', 'GarageArea']

X1 = home_data[updatedFeatures]



train_X1, val_X1, train_y1, val_y1 = train_test_split(X1, y, random_state=1)

rf_model1 = RandomForestRegressor(random_state=1)

rf_model1.fit(train_X1, train_y1)

rf_val_predictions1 = rf_model1.predict(val_X1)

rf_val_mae1 = mean_absolute_error(rf_val_predictions1, val_y1) 



print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae1))
home_data.describe().columns[1:-1]
allFeatures = home_data.describe().columns[1:-1]

X2 = home_data[allFeatures].fillna(0)



train_X2, val_X2, train_y2, val_y2 = train_test_split(X2, y, random_state=1)

rf_model2 = RandomForestRegressor(random_state=1)

rf_model2.fit(train_X2, train_y2)

rf_val_predictions2 = rf_model2.predict(val_X2)

rf_val_mae2 = mean_absolute_error(rf_val_predictions2, val_y2) 



print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae2))
test_X2 = test_data[allFeatures].fillna(0)

test_preds2 = rf_model2.predict(test_X2)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': test_preds2})

print(output)

output.to_csv('submission.csv', index=False)