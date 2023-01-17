#just some imports =)

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np



from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.impute import SimpleImputer

from xgboost import XGBRegressor



from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence

home_data = pd.read_csv('../input/train.csv') #reading data

home_data.head()
numeric_features = home_data.select_dtypes(include=[np.number]) #getting only numeric values

corr = numeric_features.corr() #calculating correlation

print(corr['SalePrice'].sort_values(ascending=False)[1:11], '\n') #getting top 10 correlations except the target

print(corr['SalePrice'].sort_values(ascending=False)[-1:-11:-1], '\n') #getting top 10 correlations except the target
features = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd'] #defininig features

y = home_data.SalePrice #target

X = home_data[features] #dataset

X.head()
cols_with_missing = [col for col in X.columns if X[col].isnull().any()] # there is no empty values =)

print(cols_with_missing)

print(X.dtypes) #all atributes are numeric. We dont need to encoding this atributes
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1) #spliting data
iowa_model_dt = DecisionTreeRegressor(random_state=1) # Specify Model

iowa_model_dt.fit(train_X, train_y) # Fit Model



# Make validation predictions and calculate mean absolute error

val_predictions = iowa_model_dt.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))
# Define the model. Set random_state to 1

rf_model = RandomForestRegressor(random_state=1)

rf_model.fit(train_X, train_y)

rf_val_predictions = rf_model.predict(val_X)

rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)



print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))
xg_model = XGBRegressor()

xg_model.fit(train_X, train_y, verbose=False)

xg_val_predictions = xg_model.predict(val_X)

xg_val_mae = mean_absolute_error(xg_val_predictions, val_y)



print("Validation MAE for XGBoost Model: {:,.0f}".format(xg_val_mae))
xg_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)

xg_model.fit(train_X, train_y, early_stopping_rounds=5, 

              eval_set=[(val_X, val_y)], verbose=True)

xg_val_predictions = xg_model.predict(val_X)

xg_val_mae = mean_absolute_error(xg_val_predictions, val_y)



print("Validation MAE for 2nd XGBoost Model: {:,.0f}".format(xg_val_mae))
from sklearn.ensemble import GradientBoostingRegressor

xg_plot_model = GradientBoostingRegressor()

xg_plot_model.fit(train_X, train_y)

plots = plot_partial_dependence(xg_plot_model, features=[0,1], X=train_X, feature_names = ['OverallQual', 'GrLivArea'])
plots = plot_partial_dependence(xg_plot_model, features=[0,1], X=train_X, feature_names = ['GarageCars', 'GarageArea'], grid_resolution=10)
plots = plot_partial_dependence(xg_plot_model, features=[0,1], X=train_X, feature_names = ['TotalBsmtSF', '1stFlrSF'], grid_resolution=10)
#training with all data

#rf_model_full_data = RandomForestRegressor(random_state=1)

#rf_model_on_full_data.fit(X, y)



xg_full = XGBRegressor()

xg_full.fit(X, y)
# read test data file using pandas

test_data = pd.read_csv('../input/test.csv')

test_X = test_data[features]

test_X.head()
#cols_with_missing = [col for col in X.columns if test_X[col].isnull().any()] # there is empty values =(

#print(cols_with_missing)

#

#my_imputer = SimpleImputer()

#test_X = my_imputer.fit_transform(test_X)
# make predictions which we will submit. 

test_preds = xg_full.predict(test_X)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.

output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': test_preds})

output.to_csv('submission0.csv', index=False)



