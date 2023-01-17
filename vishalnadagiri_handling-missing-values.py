# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# print(os.listdir("../input/melb_data.csv'"))
file_path = '../input/melb_data.csv'

home_data = pd.read_csv(file_path)

# Any results you write to the current directory are saved as output.
#getting required functions
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
home_data.head(2)
melb_target = home_data.Price
melb_predictors = home_data.drop(['Price'], axis=1)
print(melb_target.head())
print('='*50)
print(melb_predictors.head())

#use only numeric predictors. 
melb_numeric_predictors = melb_predictors.select_dtypes(exclude=['object'])
#Split the data set
X_train, X_test, y_train, y_test = train_test_split(melb_numeric_predictors, 
                                                    melb_target,
                                                    train_size=0.7, 
                                                    test_size=0.3, 
                                                    random_state=0)
#function score_dataset o compare the quality of diffrent approaches to missing values. 
#This function reports the out-of-sample MAE score from a RandomForest.

def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)
#get a column names with mssng values
cols_with_missing = [col for col in X_train.columns 
                                 if X_train[col].isnull().any()]
cols_with_missing
#drop a coumns wth missing value and make a new tran and test data
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_test  = X_test.drop(cols_with_missing, axis=1)
print('origional training set {}'.format(X_train.shape))
print('reduced training set {}' .format(reduced_X_train.shape))
print('='*10)
print('origional test set {}' .format(X_test.shape))
print('reduced testing set {}' .format(reduced_X_test.shape))
#MAE from dropping columns with missing alues

print("Mean Absolute Error from dropping columns with Missing Values:")
print(score_dataset(reduced_X_train, reduced_X_test, y_train, y_test))
#MAE from imputation of columns with missing alues
from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()
imputed_X_train = my_imputer.fit_transform(X_train)
imputed_X_test = my_imputer.transform(X_test)
print("Mean Absolute Error from Imputation:")
print(score_dataset(imputed_X_train, imputed_X_test, y_train, y_test))
#observe the error difference. That is error is reduced by
print('error difference {}' 
      .format(score_dataset(reduced_X_train, reduced_X_test, y_train, y_test)-score_dataset(imputed_X_train, imputed_X_test, y_train, y_test)))
#copy the data set
imputed_X_train_plus = X_train.copy()
imputed_X_test_plus = X_test.copy()

#get the columns with missing values 
cols_with_missing = (col for col in X_train.columns 
                                 if X_train[col].isnull().any())

for col in cols_with_missing:
    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()
    imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()
    
    
#Imputation
my_imputer = SimpleImputer()
imputed_X_train_plus = my_imputer.fit_transform(imputed_X_train_plus)
imputed_X_test_plus = my_imputer.transform(imputed_X_test_plus)

print("Mean Absolute Error from Imputation while Track What Was Imputed:")
print(score_dataset(imputed_X_train_plus, imputed_X_test_plus, y_train, y_test))
