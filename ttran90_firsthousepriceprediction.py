# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# Any results you write to the current directory are saved as output.
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

y = train.SalePrice
X = train.drop(['Id','SalePrice'], axis=1)
val_X = test.drop(['Id'], axis=1)

#split the training set so that estimated MAE can be computed
numeric_predictors = X.select_dtypes(exclude=['object'])
X_train, X_test, y_train, y_test = train_test_split(numeric_predictors, 
                                                    y,
                                                    train_size=0.7, 
                                                    test_size=0.3, 
                                                    random_state=0)

#compute MAE
def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)

new_train = X_train.copy()
new_test = X_test.copy()
col_missing = [col for col in new_train.columns if new_train[col].isnull().any()]

for col in col_missing:
    new_train[col + '_missing'] = new_train[col].isnull() 
    new_test[col + '_missing'] = new_test[col].isnull()
    
my_imputer = SimpleImputer()
new_train = my_imputer.fit_transform(new_train)
new_test = my_imputer.transform(new_test)

print("Mean Absolute Error from Imputation while Track What Was Imputed:")
print(score_dataset(new_train, new_test, y_train, y_test))

# my_model = RandomForestRegressor()
# my_model.fit(X, y)
# predicted = my_model.predict(val_X)
# print(predicted)
#categorical data
#use OHE before imputation
one_hot_encoded_train = pd.get_dummies(X)
one_hot_encoded_test = pd.get_dummies(val_X)
final_train_ohe, final_test_ohe = one_hot_encoded_train.align(one_hot_encoded_test,
                                                                    join='inner', 
                                                                    axis=1)

#Train the model when categorical data are transformed in to numerical data
X_num = final_train_ohe.select_dtypes(exclude=['object'])
val_X_num = final_test_ohe.select_dtypes(exclude=['object'])
final_train = X_num.copy()
final_test = val_X_num.copy()

#Imputation
col_missing = [col for col in final_train.columns if final_train[col].isnull().any()]

for col in col_missing:
    final_train[col + '_missing'] = final_train[col].isnull() 
    final_test[col + '_missing'] = final_test[col].isnull()

#numerical data
my_imputer = SimpleImputer()
final_train = my_imputer.fit_transform(final_train)
final_test = my_imputer.transform(final_test)


my_model = RandomForestRegressor()
my_model.fit(final_train, y)
predicted = my_model.predict(final_test)
print(predicted)

my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)