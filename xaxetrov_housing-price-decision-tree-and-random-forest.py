import graphviz



import pandas as pd



from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence

from sklearn.impute import SimpleImputer

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor, export_graphviz
file_path = '../input/train.csv'

home_data = pd.read_csv(file_path)



home_data.describe(include='number')
home_data.describe(include='object')
# Create target object and call it y

y = home_data.SalePrice

# Create X

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = home_data[features]

X.describe()
# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
# Create and fit model using ok value for max_leaf_nodes (100 is better)

tree_model = DecisionTreeRegressor(max_leaf_nodes=20, random_state=1)

tree_model.fit(train_X, train_y)



# Make predictions on validation data

val_predictions = tree_model.predict(val_X)



# Evaluate predictions

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))

print("Related information:\n  - mean: {:,.0f}\n  - 50% value: {:,.0f}".format(y.mean(), y.median()))
tree_graph = export_graphviz(

    tree_model,

    out_file=None,

    feature_names=features,

    rounded = True,

    filled = True

)

graphviz.Source(tree_graph)
file_path = '../input/train.csv'

home_data = pd.read_csv(file_path)

# Create target object and call it y

y = home_data.SalePrice

# Create X

X = home_data.drop('SalePrice', 1)
home_data.describe(include='number')
home_data.describe(include='object')
# Dummification



dummied_X = pd.get_dummies(X)



# Imputation wit extra column



# Get columns that have missing values

X_plus = dummied_X.copy()

cols_with_missing = (col for col in X_plus.columns 

                                 if X_plus[col].isnull().any())

# Mark missing values by adding a column for them

for col in cols_with_missing:

    X_plus[col + '_was_missing'] = X_plus[col].isnull()



# Call imputer

my_imputer = SimpleImputer()

imputed_X_plus = my_imputer.fit_transform(X_plus)
# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(imputed_X_plus, y, random_state=1)


# Fit model

rf_model = RandomForestRegressor(n_estimators = 200, random_state=1)

rf_model.fit(train_X, train_y)

rf_val_predictions = rf_model.predict(val_X)

rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)



print("Mean Absolute Error with imputation (tracking imputed values): {:,.0f}".format(rf_val_mae))
# To improve accuracy, create a new Random Forest model which we will train on all training data

rf_model_on_full_data = RandomForestRegressor(n_estimators = 200, random_state=1)



# fit rf_model_on_full_data on all data from the training data

rf_model_on_full_data.fit(imputed_X_plus, y)
# path to file you will use for predictions

test_data_path = '../input/test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)

test_X = pd.get_dummies(test_data)



# add missing columns due to incomplete dummyfication

missing_cols = set(X_plus.columns) - set(test_X.columns)

for c in missing_cols:

    test_X[c] = 0



# Imputation with extra column

imputed_test_X_plus = my_imputer.transform(test_X)



# Make predictions which we will submit. 

test_preds = rf_model_on_full_data.predict(imputed_test_X_plus)



# Save predictions in format used for competition scoring

output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)