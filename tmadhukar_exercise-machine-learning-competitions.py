# Code you have previously used to load data

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.impute import SimpleImputer

from pandas import DataFrame

# Path of the file to read. We changed the directory structure to simplify submitting to a competition

iowa_file_path = '../input/train.csv'

home_data = pd.read_csv(iowa_file_path)

home_data.dropna(axis=0, subset=['SalePrice'], inplace=True)

home_data.columns
def get_low_correlated_cols(df):

    corr = df.corr()['SalePrice']

    low_corrs = corr[corr < 0.3]

    return [col for col in low_corrs.index if col != 'Id' and col != 'SalePrice']



low_corr_columns = get_low_correlated_cols(home_data)

low_corr_columns
y = home_data.SalePrice

X = home_data.drop(['Id', 'SalePrice'] + low_corr_columns, axis=1)

print(f"Training data: X{X.shape}, y{y.shape}")

# path to file you will use for predictions

test_data_path = '../input/test.csv'

test_data = pd.read_csv(test_data_path)



test_X = test_data.drop(['Id'] + low_corr_columns, axis=1)

print(f"Test data: {test_X.shape}")
%%markdown



# Deal with categorical values
dummified_X = pd.get_dummies(X)

dummified_test_X = pd.get_dummies(test_X)

final_X, final_test_X = dummified_X.align(dummified_test_X, join='left', axis=1)

print(f"final_X{final_X.shape}, final_test_X{final_test_X.shape}")
def get_object_cols(df):

    return [col for col in df if df[col].dtype.name == 'object']



print(f" final_X: {get_object_cols(final_X)}, final_test_X: {get_object_cols(final_test_X)}")
%%markdown



# Deal with missing values
imputer = SimpleImputer()

final_imputed_X = DataFrame.from_records(imputer.fit_transform(final_X))

final_imputed_test_X = DataFrame.from_records(imputer.transform(final_test_X))



print(f"final_imputed_X{final_imputed_X.shape}, final_imputed_test_X{final_imputed_test_X.shape}")
def get_missing_cols(df):

    return [col for col in df.columns if df[col].isna().any()]



print(f"final_imputed_X: {get_missing_cols(final_imputed_X)}, final_imputed_test_X: {get_missing_cols(final_imputed_test_X)}")
%%markdown



# Train the model
from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split



train_X, val_X, train_y, val_y = train_test_split(final_imputed_X, y, test_size=0.3, random_state=2)



# model = XGBRegressor()

# model.fit(train_X, train_y)



model = XGBRegressor(n_estimators=1000, learning_rate=0.05)

model.fit(train_X, train_y, early_stopping_rounds=5, eval_set=[(train_X, train_y), (val_X, val_y)], verbose=False)

preds_y = model.predict(val_X)



from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(val_y, preds_y))
eval_result = model.evals_result()

training_rounds = range(len(eval_result['validation_0']['rmse']))



import matplotlib.pyplot as plt



plt.scatter(x=training_rounds, y=eval_result['validation_0']['rmse'], label='Training Error')

plt.scatter(x=training_rounds, y=eval_result['validation_1']['rmse'], label='Validation Error')

plt.grid(True)

plt.xlabel('Iterations')

plt.ylabel('RMSE')

plt.title('Training vs Validation Errors')

plt.legend()


# fit on the entire data set

model = XGBRegressor(n_estimators=1000, learning_rate=0.05)

model.fit(final_imputed_X, y)



# make predictions which we will submit. 

test_preds = model.predict(final_imputed_test_X)



# The lines below shows you how to save your data in the format needed to score it in the competition

output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': test_preds})

print(output)

output.to_csv('submission.csv', index=False)