# Code you have previously used to load data
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# Path of the file to read. We changed the directory structure to simplify submitting to a competition
iowa_file_path = '../input/train.csv'
#test_data_path = '../input/test.csv'
home_data = pd.read_csv(iowa_file_path)
#test_data = pd.read_csv(test_data_path)

# Overview of data
print(home_data.shape)
cols_with_missing = [col for col in home_data.columns if home_data[col].isnull().any()]
print(cols_with_missing)
home_data[cols_with_missing].info()
missing_val_count_by_column = (home_data.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column >0].sort_values(ascending = False))

# Fill missing values with none
col_fill_none = ['Alley', 'Fence', 'MasVnrType', 'MasVnrArea', 
                 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                 'FireplaceQu','GarageType','GarageYrBlt','GarageFinish','GarageQual', 'GarageCond',
                 'PoolQC', 'Fence', 'Electrical']
for col in col_fill_none:
    home_data[col].fillna('None', inplace = True)

# Fill missing values with none
col_fill_none = ['Alley', 'Fence', 'MasVnrType', 'MasVnrArea', 
                 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                 'FireplaceQu','GarageType','GarageYrBlt','GarageFinish','GarageQual', 'GarageCond',
                 'PoolQC', 'Fence', 'Electrical']
for col in col_fill_none:
    home_data[col].fillna('None', inplace = True)

# Fill missing values with imputer
LotFrontage_mean = home_data['LotFrontage'].mean()
home_data['LotFrontage'].fillna(LotFrontage_mean, inplace = True)

# Data Exploration
home_predictors = home_data.drop(['SalePrice'], axis = 1)
cols_quantity = [col for col in home_predictors.columns if home_predictors.dtypes[col] != 'object']
cols_quality = [col for col in home_predictors.columns if home_predictors.dtypes[col] == 'object']
print(len(cols_quantity)) # 35
print(len(cols_quality))  # 45

high_cardinality_cols = [cname for cname in home_predictors.columns if 
                                home_predictors[cname].nunique() > 10 and
                                home_predictors[cname].dtype == "object"]
print(len(high_cardinality_cols)) # 5

## Here can do more data exploration. Later come back!
#print(test_data.shape)
#test_missing_val_count_by_column = (test_data.isnull().sum())
#print(test_missing_val_count_by_column[test_missing_val_count_by_column >0].sort_values(ascending = False))

low_cardinality_cols = [cname for cname in home_predictors.columns if 
                                home_predictors[cname].nunique() < 10 and
                                home_predictors[cname].dtype == "object"]
print(len(low_cardinality_cols))

my_predictors = home_predictors.drop(high_cardinality_cols, axis = 1)
my_predictors_encoded = pd.get_dummies(my_predictors)
# Create target object and call it y
y = home_data.SalePrice
# Create X
#features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
#X = home_data[features]
X = my_predictors_encoded

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

xgb_model = XGBRegressor(random_state = 1)
xgb_model.fit(train_X, train_y)
xgb_val_predictions = xgb_model.predict(val_X)
xgb_val_mae = mean_absolute_error(xgb_val_predictions, val_y)
print("Validation MAE forXGboost Model: {:,.0f}".format(xgb_val_mae))
# To improve accuracy, create a new xgboost model which you will train on all training data
xgboost_model_on_full_data = XGBRegressor(random_state = 1)
# fit rf_model_on_full_data on all data from the 
#rf_model_on_full_data.fit(X,y)
xgboost_model_on_full_data.fit(X, y)
import numpy as np
# path to file you will use for predictions
test_data_path = '../input/test.csv'

# read test data file using pandas
test_data = pd.read_csv(test_data_path)
test_predictors = test_data.drop(high_cardinality_cols, axis = 1)
test_cols_with_missing = test_predictors.columns[test_predictors.isnull().any()]

#test_cols_with_missing = [col for col in test_predictors.columns if test_predictors[col].isnull().any()]
print(len(test_cols_with_missing))

test_missing_val_count_by_column = (test_predictors.isnull().sum())
print(test_missing_val_count_by_column[test_missing_val_count_by_column >0].sort_values(ascending = False))

# Fill the missing values
# dtype: float64
LotFrontage_mean = test_predictors['LotFrontage'].mean()
test_predictors['LotFrontage'].fillna(LotFrontage_mean, inplace = True)

#test_missing_numeric_cols = ['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF',
#                             'GarageArea', 'BsmtFullBath','BsmtHalfBath', 'GarageCars']
#for col in test_missing_numeric_cols:
#    test_predictors[col].fillna(test_predictors[col].mean(), inplace=True)
test_predictors['BsmtFinSF1'].fillna(test_predictors['BsmtFinSF1'].mean(), inplace = True)
test_predictors['BsmtFinSF2'].fillna(test_predictors['BsmtFinSF2'].mean(), inplace = True)
test_predictors['BsmtUnfSF'].fillna(test_predictors['BsmtUnfSF'].mean(), inplace = True)
test_predictors['TotalBsmtSF'].fillna(test_predictors['TotalBsmtSF'].mean(), inplace = True)
test_predictors['BsmtFullBath'].fillna(0, inplace = True)
test_predictors['BsmtHalfBath'].fillna(0, inplace = True)
test_predictors['GarageCars'].fillna(0, inplace = True)
test_predictors['GarageArea'].fillna(test_predictors['GarageArea'].mean(), inplace = True)

test_missing_categorical_cols = ['MSZoning','Alley','Utilities','MasVnrType','BsmtQual','BsmtCond','BsmtExposure',
                     'BsmtFinType1','BsmtFinType2','KitchenQual','Functional','FireplaceQu','GarageType',
                     'GarageFinish','GarageQual','GarageCond','PoolQC','Fence','MiscFeature','SaleType']

for col in test_missing_categorical_cols:
    test_predictors[col].fillna('None', inplace = True)


# create test_X which comes from test_data but includes only the columns you used for prediction.
# The list of columns is stored in a variable called features
#test_X = test_data[features]
test_predictors_encoded = pd.get_dummies(test_predictors)

my_predictors_encoded = pd.get_dummies(my_predictors)
final_train, final_test = my_predictors_encoded.align(test_predictors_encoded,
                                                                    join='left', 
                                                                    axis=1)
final_test_missing_col = final_test.columns[final_test.isnull().any()]
final_test[final_test_missing_col].info()
final_test.fillna(value = 0, inplace = True)


xgboost_model_on_full_data.fit(final_train, y)

#make predictions which we will submit. 

xgboost_test_preds = xgboost_model_on_full_data.predict(final_test)
# The lines below shows you how to save your data in the format needed to score it in the competition
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': xgboost_test_preds})

output.to_csv('submission.csv', index=False)