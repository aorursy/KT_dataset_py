# Code you have previously used to load data

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from learntools.core import *

from sklearn.naive_bayes import GaussianNB

from sklearn import svm

from sklearn.neighbors.nearest_centroid import NearestCentroid

from sklearn.neighbors import (NeighborhoodComponentsAnalysis,KNeighborsClassifier)





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

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.tree import DecisionTreeRegressor

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score





# Data load

sd = pd.read_csv('../input/train.csv')

sd.head() # To head first 5 rows from data set

# Preapering For Data

y = sd.SalePrice

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'SaleCondition']

X = sd[features]

print(type(X))

print("Data befor droping\n", sd.head())

print("Data After droping",X.head())

# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# Bulit machine learinig modle by sklearn

modle = GaussianNB()

modle_1 = DecisionTreeRegressor(random_state=8)

modle_2 = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)

#Fitting for data

modle.fit(train_X, train_y)

modle_1.fit(train_X, train_y)

modle_2.fit(train_X, train_y)

# Make validation predictions and calculate mean absolute error

val_predictions = modle.predict(val_X)

val_predictions_1 = modle_1.predict(val_X)

val_predictions_2 = modle_2.predict(val_X)

print('The ruslt by use Naive_baytes\n', val_predictions)

print('The ruslt by use DecisionTreeRegressor\n', val_predictions_1)

print('The ruslt by use DecisionForstRegressor\n', val_predictions_2)

val_mae = mean_absolute_error(val_predictions, val_y)

val_mae_1 = mean_absolute_error(val_predictions_1, val_y)

val_mae_2 = mean_absolute_error(val_predictions_2, val_y)

print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}\n".format(val_mae))

print("Validation MAE when not specifying max_leaf_nodes by DecisionTreeRegressor: {:,.0f}\n".format(val_mae_1))

print("Validation MAE when not specifying max_leaf_nodes by DecisionForstRegressor: {:,.0f}\n".format(val_mae_2))

print("\n")

# Evaluatin for data

score = accuracy_score(y_true=val_y, y_pred=val_predictions)

score_1 = accuracy_score(y_true=val_y, y_pred=val_predictions_1)

#score_2 = accuracy_score(y_true=val_y, y_pred=val_predictions_2)

print('accuracy_score from Naive\n', score)

print('accuracy_score from DecisionTreeRegressor\n', score_1)

#print('accuracy_score from DecisionForstRegressor', score_2)
# Data load

import pandas as pd

from sklearn.metrics import accuracy_score

sd = pd.read_csv('../input/train.csv')

sd.head() # To head first 5 rows from data set

# Preapering For Data

y = sd.SalePrice

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'SaleCondition','SaleType']

X = sd[features]

print(type(X))

print("Data befor droping\n", sd.head())

print("Data After droping",X.head())

#bulit modle

from sklearn.ensemble import RandomForestRegressor

xs = RandomForestRegressor(random_state=5)

xs.fit(X, y)

preds = xs.predict(X)

print(preds)

score_2 = accuracy_score(y_true=y, y_pred=preds)

print('accuracy_score from DecisionForstRegressor', score_2)
# To improve accuracy, create a new Random Forest model which you will train on all training data

rf_model_on_full_data = RandomForestRegressor()

#rf_model_on_full_data = GaussianNB()

#rf_model_on_full_data = DecisionTreeRegressor(random_state=8)

#rf_model_on_full_data = svm.SVR()

#rf_model_on_full_data = NearestCentroid()

#rf_model_on_full_data = KNeighborsClassifier(n_neighbors=3)



# fit rf_model_on_full_data on all data from the training data

rf_model_on_full_data.fit(X, y)

test_data_path = '../input/test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)

test_data.head()

test_data.columns
# path to file you will use for predictions

test_data_path = '../input/test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)



# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

test_X = test_data[features] 

#Fiting data

rf_model_on_full_data.fit(X, y)



# make predictions which we will submit. 

test_preds = rf_model_on_full_data.predict(test_X )

print(test_preds)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': test_data.Id,

                      'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)