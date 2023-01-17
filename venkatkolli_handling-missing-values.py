import pandas as pd
# Load Data
iowa_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
print(iowa_data.columns)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
iowa_target=iowa_data['SalePrice']
iowa_predictors=iowa_data.drop(['SalePrice'], axis=1)
# For the sake of keeping the example simple, we'll use only numeric predictors.
iowa_num_predictors = iowa_predictors.select_dtypes(exclude=['object'])
x_train, x_test, y_train, y_test = train_test_split(iowa_num_predictors, iowa_target, train_size=0.7, test_size=0.3, random_state=0)
def score_dataset(x_train, x_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    return mean_absolute_error (y_test, preds)
cols_with_missing = [col for col in x_train.columns
                                if x_train[col].isnull().any()]
reduced_x_train = x_train.drop(cols_with_missing, axis = 1)
reduced_x_test = x_test.drop(cols_with_missing, axis = 1)
print("Mean Absolute Error from dropping columns with Missing Values:")
print(score_dataset(reduced_x_train, reduced_x_test, y_train, y_test))
from sklearn.preprocessing import Imputer

my_imputer = Imputer()
imputed_x_train = my_imputer.fit_transform(x_train)
imputed_x_test = my_imputer.transform(x_test)
print("Mean Absolute Error from Imputation:")
print(score_dataset(imputed_x_train,imputed_x_test, y_train, y_test))
imputed_x_train_plus = x_train.copy()
imputed_x_test_plus = x_test.copy()

cols_with_missing = (col for col in x_train.columns 
                                 if x_train[col].isnull().any())
for col in cols_with_missing:
    imputed_x_train_plus[col + '_was_missing'] = imputed_x_train_plus[col].isnull()
    imputed_x_test_plus[col + '_was_missing'] = imputed_x_test_plus[col].isnull()

# Imputation
my_imputer = Imputer()
imputed_x_train_plus = my_imputer.fit_transform(imputed_x_train_plus)
imputed_x_test_plus = my_imputer.transform(imputed_x_test_plus)

print("Mean Absolute Error from Imputation while Track What Was Imputed:")
print(score_dataset(imputed_x_train_plus, imputed_x_test_plus, y_train, y_test))