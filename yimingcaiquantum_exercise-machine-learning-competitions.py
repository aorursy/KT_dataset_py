# Code you have previously used to load data
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

# Path of the file to read. We changed the directory structure to simplify submitting to a competition
iowa_file_path = '../input/train.csv'

home_data = pd.read_csv(iowa_file_path)
# Create target object and call it y
y = home_data.SalePrice
# Create X
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd'
            , 'OverallQual', 'OverallCond', 'YearRemodAdd','TotalBsmtSF', 'GrLivArea','KitchenAbvGr', 'GarageArea',  'PoolArea', 'YrSold'
            , 'LotConfig', 'Neighborhood', 'BldgType', 'HouseStyle','ExterQual','BsmtQual','HeatingQC', 'KitchenQual'
            , 'GarageType', 'GarageFinish', 'PavedDrive', 'SaleType', 'SaleCondition']
X = home_data[features]

#one hot key convert non-integer data
X_one_hot = pd.get_dummies(X)

#imputation
X_imputed = X_one_hot.copy()
#cols_with_missing = [col for col in X_one_hot.columns if X_one_hot[col].isnull().any()]
#for col in cols_with_missing:
#    X_imputed[col+'_was_missing'] = X_imputed[col].isnull()
imputer = SimpleImputer()
X_imputed = imputer.fit_transform(X_imputed)
    
# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X_imputed, y, random_state=1)
# # use XGBoost model
# iowa_model = XGBRegressor()
# # Fit Model
# iowa_model.fit(train_X, train_y)
# the naive decision tree Model
iowa_model = DecisionTreeRegressor(random_state=1)
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

# Use random forest model. Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

# use naive XGBoost model
xgb_model = XGBRegressor()
xgb_model.fit(train_X, train_y)
xgb_val_predictions = xgb_model.predict(val_X)
xgb_val_mae = mean_absolute_error(xgb_val_predictions, val_y)
print("Validation MAE for XGBoost Model: {:,.0f}".format(xgb_val_mae))

# use naive XGBoost model with large n_estimator and early stopping
xgb_model2 = XGBRegressor(n_estimator = 1000, learning_rate = 0.1)
xgb_model2.fit(train_X, train_y, early_stopping_rounds = 10,
             eval_set = [(val_X, val_y)], verbose = False)
xgb_val_predictions2 = xgb_model2.predict(val_X)
xgb_val_mae2 = mean_absolute_error(xgb_val_predictions2, val_y)
print("Validation MAE for XGBoost Model with n_est: {:,.0f}".format(xgb_val_mae2))
for nes in [500, 1000, 1500, 2000]:
    for lr in [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        xgb_model2 = XGBRegressor(n_estimator = 1000, learning_rate = lr)
        xgb_model2.fit(train_X, train_y, early_stopping_rounds = 10,
                     eval_set = [(val_X, val_y)], verbose = False)
        xgb_val_predictions2 = xgb_model2.predict(val_X)
        xgb_val_mae2 = mean_absolute_error(xgb_val_predictions2, val_y)
        print(str(nes) + "," + str(lr) + "Validation MAE for XGBoost Model with n_est: {:,.0f}".format(xgb_val_mae2))
#for lr in [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]:
#for lr in [0.1,0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3]:
#for lr in range(240,260):
for lr in range(2500,2520):
    xgb_model2 = XGBRegressor(n_estimator = 1000, learning_rate = 0.0001*lr)
    xgb_model2.fit(train_X, train_y, early_stopping_rounds = 10,
                 eval_set = [(val_X, val_y)], verbose = False)
    xgb_val_predictions2 = xgb_model2.predict(val_X)
    xgb_val_mae2 = mean_absolute_error(xgb_val_predictions2, val_y)
    print(str(0.0001*lr) + "Validation MAE for XGBoost Model with n_est: {:,.0f}".format(xgb_val_mae2))
# To improve accuracy, create a new Random Forest model which you will train on all training data
#rf_model_on_full_data = RandomForestRegressor()

# fit rf_model_on_full_data on all data from the 
#rf_model_on_full_data.fit(train_X, train_y)
#rf_model_on_full_data.fit(X_imputed, y)

# what's the purpose of this block??
# answer: in previous block we split the home_data into train and test,
# but actually, all home_data is train data, and the real test data 
# is in next block.

#print(len(X_one_hot.dtypes))
#print(len(X_one_hot.columns))
# path to file you will use for predictions
test_data_path = '../input/test.csv'

# read test data file using pandas
test_data = pd.read_csv(test_data_path)

# create test_X which comes from test_data but includes only the columns you used for prediction.
# The list of columns is stored in a variable called features
test_X = test_data[features]
# one hot key to turn object into int
test_X_one_hot = pd.get_dummies(test_X)
#X_one_hot, test_X_one_hot = X_one_hot.align(test_X_one_hot, join = 'left', axis = 1)
final_train, final_test  = X_one_hot.align(test_X_one_hot, join = 'inner', axis = 1)

#print(test_X_one_hot.dtypes)
# print(len(test_X_one_hot.columns))
# print(len(final_test.columns))
# print(len(final_train.columns))
# impute test data
test_X_imputed = final_test.copy()
train_X_imputed = final_train.copy()

imputer = SimpleImputer()
train_X_imputed = imputer.fit_transform(train_X_imputed)        
test_X_imputed = imputer.fit_transform(test_X_imputed)

#test_cols_with_missing = [col for col in final_test.columns if final_test[col].isnull().any()]
#for col in test_cols_with_missing:
#    test_X_imputed[col+'_was_missing'] = test_X_imputed[col].isnull()
#print(len(test_X_imputed.columns))
# if the test set has missing value in different columns as train set, the number of columns will not match. 

#train model
#rf_model_on_full_data.fit(train_X_imputed, y)
xgb_model_final = XGBRegressor(n_estimator = 1000, learning_rate = 0.25)
xgb_model_final.fit(train_X_imputed, y, early_stopping_rounds = 10, eval_set = [(train_X_imputed, y)], verbose = False)
#xgb_model_final.fit(train_X_imputed, y, early_stopping_rounds = 10, verbose = False)
# make predictions which we will submit. 
#test_preds = rf_model_on_full_data.predict(test_X_imputed)
test_preds = xgb_model_final.predict(test_X_imputed)

# The lines below shows you how to save your data in the format needed to score it in the competition
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)