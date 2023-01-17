import pandas as pd



# Load data

melb_data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')



from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split



melb_target = melb_data.Price

melb_predictors = melb_data.drop(['Price'], axis=1)



print(melb_target.head().tolist())

print(melb_predictors.head())

# For the sake of keeping the example simple, we'll use only numeric predictors. 

melb_numeric_predictors = melb_predictors.select_dtypes(exclude=['object'])

print(melb_numeric_predictors.head())
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(melb_numeric_predictors, 

                                                    melb_target,

                                                    train_size=0.7, 

                                                    test_size=0.3, 

                                                    random_state=0)

def score_dataset(X_train, X_test, y_train, y_test):

    model = RandomForestRegressor()

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    return mean_absolute_error(y_test, preds)
cols_with_missing = [col for col in X_train.columns 

                                 if X_train[col].isnull().any()]

reduced_X_train = X_train.drop(cols_with_missing, axis=1)

reduced_X_test  = X_test.drop(cols_with_missing, axis=1)

print("Mean Absolute Error from dropping columns with Missing Values:")

print(score_dataset(reduced_X_train, reduced_X_test, y_train, y_test))
from sklearn.impute import SimpleImputer



my_imputer = SimpleImputer()

imputed_X_train = my_imputer.fit_transform(X_train)

imputed_X_test = my_imputer.transform(X_test)

print("Mean Absolute Error from Imputation:")

print(score_dataset(imputed_X_train, imputed_X_test, y_train, y_test))
# X_train columns

print(X_train.columns)



#find miss columns

cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]

print(cols_with_missing)



#print(X_train.describe())
imputed_X_train_plus = X_train.copy()

imputed_X_test_plus = X_test.copy()



cols_with_missing = (col for col in X_train.columns 

                                 if X_train[col].isnull().any())

for col in cols_with_missing:

    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()

    imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()

    

#print(imputed_X_train_plus.isnull())

#print(imputed_X_train_plus.isnull().any())

#print(imputed_X_train_plus['Car_was_missing'].count())

#print(imputed_X_train_plus.isnull().sum())

#print(imputed_X_test_plus['Car'].isnull())

#print(imputed_X_test_plus['Car'])

#print(imputed_X_test_plus['Rooms'].isnull())



# Imputation

my_imputer = SimpleImputer()

imputed_X_train_plus = my_imputer.fit_transform(imputed_X_train_plus)

imputed_X_test_plus = my_imputer.transform(imputed_X_test_plus)



#print(pd.DataFrame(imputed_X_train_plus).isnull().any())



print("Mean Absolute Error from Imputation while Track What Was Imputed:")

print(score_dataset(imputed_X_train_plus, imputed_X_test_plus, y_train, y_test))