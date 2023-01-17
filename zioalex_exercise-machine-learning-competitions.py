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

print(f"Target lenght: {len(y)}")

# To improve accuracy, create a new Random Forest model which you will train on all training data

rf_model_on_full_data = RandomForestRegressor(random_state=1)



# fit rf_model_on_full_data on all data from the training data

rf_model_on_full_data.fit(X,y)

# path to file you will use for predictions

test_data_path = '../input/test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)





# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features

# features = ['MSSubClass', 'Neighborhood', 'HouseStyle', 'OverallQual', 'OverallCond', 'LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']  



# features = ['MSSubClass', 'Neighborhood_num', 'HouseStyle_num', 'OverallQual', 'OverallCond', 'LotArea', 'YearBuilt', '1stFlrSF', 

#             '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']  



test_X = test_data[features]

print(test_X.head())

# make predictions which we will submit. 

test_preds = rf_model_on_full_data.predict(test_X)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': test_data.Id, 

                       'SalePrice': test_preds})

# output.to_csv('submission.csv', index=False)

output.head()
# path to file you will use for predictions

test_data_path = '../input/test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)

# 'Neighborhood_num', 'HouseStyle_num'

x = 0

# Neighborhood_len = len(test_data.Neighborhood.unique())

# Neighborhood_dict = { i: x for x, i  in enumerate(test_data.Neighborhood.unique())   }

# Neighborhood_dict



def convert_to_num(data_source: pd, column: str) -> pd:

    list_dict = { i: x for x, i  in enumerate(data_source[column].unique()) }

    data_source[column + "_num"] = data_source[column].apply(list_dict.get)

    return data_source

# Add new features in test dataframe

pd_new = convert_to_num(test_data, "Neighborhood")

pd_new = convert_to_num(test_data, "HouseStyle")

pd_new = convert_to_num(test_data, "MSZoning")

pd_new = convert_to_num(test_data, "HouseStyle")

pd_new = convert_to_num(test_data, "Exterior2nd")

pd_new = convert_to_num(test_data, "ExterQual") 

pd_new = convert_to_num(test_data, "Foundation")

pd_new = convert_to_num(test_data, "BsmtFinType1")

pd_new = convert_to_num(test_data, "HeatingQC")

pd_new.describe()





            
# home_data: 1460 elements - train.csv

# test_data: 1459 elements - test.csv

# ValueError: Number of labels=1460 does not match number of samples=1459



pd_train_new = convert_to_num(home_data, "Neighborhood")

pd_train_new = convert_to_num(home_data, "HouseStyle")

pd_train_new = convert_to_num(home_data, "MSZoning")

pd_train_new = convert_to_num(home_data, "HouseStyle")

pd_train_new = convert_to_num(home_data, "Exterior2nd") 

pd_train_new = convert_to_num(home_data, "ExterQual")

pd_train_new = convert_to_num(home_data, "Foundation")

pd_train_new = convert_to_num(home_data, "BsmtFinType1")

pd_train_new = convert_to_num(home_data, "HeatingQC")







features_add = ['MSSubClass', 'Neighborhood_num', 'HouseStyle_num', 'OverallQual', 'OverallCond', 'LotArea', 'YearBuilt',

                '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'MSZoning_num','HouseStyle_num',

                'YearRemodAdd','Exterior2nd_num', 'ExterQual_num', 'Foundation_num', 'BsmtFinType1_num', 'BsmtUnfSF',

                'HeatingQC_num', 'GarageArea']

# Remove features with missing value:

no_value_features = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt' ]

features_add_clean = features_add - no
# home_data: 1460 elements - train.csv

# test_data: 1459 elements - test.csv

# ValueError: Number of labels=1460 does not match number of samples=1459



pd_train_new = convert_to_num(home_data, "Neighborhood")

pd_train_new = convert_to_num(home_data, "HouseStyle")

pd_train_new = convert_to_num(home_data, "MSZoning")

pd_train_new = convert_to_num(home_data, "HouseStyle")

pd_train_new = convert_to_num(home_data, "Exterior2nd") 

pd_train_new = convert_to_num(home_data, "ExterQual")

pd_train_new = convert_to_num(home_data, "Foundation")

pd_train_new = convert_to_num(home_data, "BsmtFinType1")

pd_train_new = convert_to_num(home_data, "HeatingQC")







features_add = ['MSSubClass', 'Neighborhood_num', 'HouseStyle_num', 'OverallQual', 'OverallCond', 'LotArea', 'YearBuilt',

                '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'MSZoning_num','HouseStyle_num',

                'YearRemodAdd','Exterior2nd_num', 'ExterQual_num', 'Foundation_num', 'BsmtFinType1_num', 'BsmtUnfSF',

                'HeatingQC_num', 'GarageArea']

# Remove features with missing value:

no_value_features = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt', 'BsmtUnfSF', 'GarageArea'  ]

features_add_clean = list(set(features_add) - set(no_value_features))

Xadd = pd_train_new[features_add_clean]



Xadd.describe()

# To improve accuracy, create a new Random Forest model which you will train on all training data

rf_model_on_full_data = RandomForestRegressor(random_state=1)



print(f"Features lenght: {len(Xadd)}\tTarget lenght: {len(y)}")



# fit rf_model_on_full_data on all data from the training data

rf_model_on_full_data.fit(Xadd,y)
# First option remove columns with empty values



cols_with_missing = [col for col in home_data.columns 

                                 if home_data[col].isnull().any()]

# You can have also missing values in the test data and therefore you may need to create another list with empty values there.



reduced_train_data = home_data.drop(cols_with_missing, axis=1)

reduced_test_data = test_data.drop(cols_with_missing, axis=1)
#test_X = pd_new[features_add]

# use features list with all the values

# test_X = pd_new[features_add_clean]

# column_with_nan = (test_X.isnull().sum())  # See https://www.kaggle.com/dansbecker/handling-missing-values

# print(column_with_nan[column_with_nan > 0])



test_X = reduced_test_data[features_add_clean]

column_with_nan = (test_X.isnull().sum())  # See https://www.kaggle.com/dansbecker/handling-missing-values

print(column_with_nan[column_with_nan > 0])





# make predictions which we will submit. 

test_preds = rf_model_on_full_data.predict(test_X)



# test_X

# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': test_data.Id, 

                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)

output.head()