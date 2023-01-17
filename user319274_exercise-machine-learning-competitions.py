# Code you have previously used to load data
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer



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

## Missing data versions
def score_predict(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)

X_predictors = home_data.drop(['SalePrice'], axis=1)
X_numeric_predictors = X_predictors.select_dtypes(exclude=['object'])

X_train, X_test, y_train, y_test = train_test_split(X_numeric_predictors, 
                                                    y,
                                                    train_size=0.7, 
                                                    test_size=0.3, 
                                                    random_state=0)

imputed_X_train_plus = X_train.copy()
imputed_X_test_plus = X_test.copy()

cols_with_missing = (col for col in X_train.columns 
                                 if X_train[col].isnull().any())
for col in cols_with_missing:
    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()
    imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()

my_imputer = SimpleImputer()
imputed_X_train_plus = my_imputer.fit_transform(imputed_X_train_plus)
imputed_X_test_plus = my_imputer.transform(imputed_X_test_plus)

print(score_predict(imputed_X_train_plus, imputed_X_test_plus, y_train, y_test))
## Missing data versions
max_leaf_nodes = [100, 200, 300, 400, 500]
for i in max_leaf_nodes:
    rf_model = RandomForestRegressor(max_leaf_nodes = i, random_state=1)
    rf_model.fit(imputed_X_train_plus, y_train)
    rf_val_predictions = rf_model.predict(imputed_X_test_plus)
    rf_val_mae = mean_absolute_error(rf_val_predictions, y_test)
    print("Validation MAE for Random Forest Model: {:,.0f}\t\t node = {}".format(rf_val_mae, i))
## Missing data versions
imputed_X_numeric_predictors_plus = X_numeric_predictors.copy()

cols_with_missing = (col for col in X_numeric_predictors.columns 
                                 if X_numeric_predictors[col].isnull().any())
for col in cols_with_missing:
    imputed_X_numeric_predictors_plus[col + '_was_missing'] = imputed_X_numeric_predictors_plus[col].isnull()

my_imputer = SimpleImputer()
imputed_X_numeric_predictors_plus = my_imputer.fit_transform(imputed_X_numeric_predictors_plus)
## Missing data versions
# To improve accuracy, create a new Random Forest model which you will train on all training data
rf_model_on_full_data = RandomForestRegressor(max_leaf_nodes = 300, random_state=1)

# fit rf_model_on_full_data on all data from the 
rf_model_on_full_data.fit(imputed_X_numeric_predictors_plus, y)

# path to file you will use for predictions
#test_data_path = '../input/test.csv'

# read test data file using pandas
#test_data = pd.read_csv(test_data_path)

# create test_X which comes from test_data but includes only the columns you used for prediction.
# The list of columns is stored in a variable called features
#test_X = test_data[features]

# make predictions which we will submit. 
#test_preds = rf_model_on_full_data.predict(test_X)

# The lines below shows you how to save your data in the format needed to score it in the competition
#output = pd.DataFrame({'Id': test_data.Id,
#                       'SalePrice': test_preds})

#output.to_csv('submission.csv', index=False)
## Missing data versions
test_data_path = '../input/test.csv'
test_data = pd.read_csv(test_data_path)
test_data = test_data.select_dtypes(exclude=['object'])

test_data_plus = test_data.copy()

test_cols_with_missing = (t_col for t_col in test_data.columns 
                                 if test_data[t_col].isnull().any())

for ccol in test_cols_with_missing:
    test_data_plus[ccol + '_was_missing'] = test_data_plus[ccol].isnull()

my_imputer = SimpleImputer()
test_data_plus = my_imputer.fit_transform(test_data_plus)

test_preds = rf_model_on_full_data.predict(test_data_plus[:,0:40])

output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)