# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 21:45:05 2018

@author: vino
"""

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # for visualization
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import Imputer
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score




'''Reading data from CSV file'''
# data_set = pd.read_csv("/home/vino/Work/Data_Sets/Hostel.csv")
data_set = pd.read_csv("../input/Hostel.csv")



'''Info about data set'''
print(data_set.info())
print(data_set.columns) # Gives all the column names.
print(data_set.shape) # Number of rows and columns.



'''********************Data Cleaning********************'''

corr_data_set = data_set
print(corr_data_set['City'].value_counts(), 
      corr_data_set['rating.band'].value_counts())



'''In corr_data_set rating.band is a categorical value and we are filling the 
missing value of it using unknown category.'''
corr_data_set['rating.band'] = corr_data_set['rating.band'].fillna('Unknown')



'''Converting categorical value to numerical attributes using label encoder.'''
label_encoder = LabelEncoder()
for key in ['rating.band', 'City']:
    print(corr_data_set[key].value_counts())
    corr_data_set[key] = label_encoder.fit_transform(corr_data_set[key])
    print(corr_data_set[key])
    
'''One hot encoding technique.'''
one_hot_encoder = OneHotEncoder()
for key in ['rating.band', 'City']:
    print(corr_data_set[key].value_counts())
    corr_data_set[key] = one_hot_encoder.fit_transform(
            (corr_data_set[key].values).reshape(-1,1))
    print(corr_data_set[key])
    
#'''Converts categorical to label encoding and one hot encoding in a single 
#                        shot'''
#label_binarizer = LabelBinarizer()
#for key in ['rating.band', 'City']:
#    print(corr_data_set[key].value_counts())
#    corr_data_set[key] = label_binarizer.fit_transform(corr_data_set[key])
#    print(corr_data_set[key].value_counts())
     
    
    
'''Removing string from distance and making it as numerical variable for linear
regression'''
print(corr_data_set['Distance'])
for data in corr_data_set['Distance']:
    temp = data.split('km')
    corr_data_set['Distance'] = temp[0]
pd.to_numeric(corr_data_set['Distance'], errors='ignore')
print(corr_data_set['Distance'])
    


'''Dropping text fields.'''
corr_data_set = corr_data_set.drop(['hostel.name', 'City', 'rating.band'], axis=1) 
print(corr_data_set.info())



'''To fill the missing values with median.'''
imputer = Imputer(strategy="median")
imputer.fit(corr_data_set)
print(imputer.statistics_) # Prints the median value for each column.
print(corr_data_set.median().values) # Prints the median value for each column.
corr_cleaned_data_array = imputer.transform(corr_data_set) # Applies median 
        # value to to each missing cell. Result is plain numpy array.
corr_cleaned_data = pd.DataFrame(corr_cleaned_data_array, 
    columns=corr_data_set.columns) # Converting array back to data frame.
print(corr_cleaned_data)                        
print(corr_cleaned_data.info())



'''Correlation'''
corr_matrix = corr_cleaned_data.corr()
print(corr_matrix)



'''Correaltion for each columns'''
for data in corr_cleaned_data.columns:
    print("*******************" +data.upper()+ "*******************")
    pd.set_option('display.precision', 20) # Used to convert exponentioal 
                                                # notation to floating point.
    print(corr_matrix[data.strip()])

'''Print correlation matrix for each feature against all feature.'''
scatter_matrix(corr_cleaned_data, figsize=(20, 20))



''' Feature and Target Variable'''
# Feature variable picked based on correlation with target variable.
feature_variable_attributes = ['summary.score', 'atmosphere', 'cleanliness', 
                               'facilities', 'location.y', 'security', 'staff']
feature_variables = corr_cleaned_data[feature_variable_attributes]
target_variable = corr_cleaned_data['valueformoney']

'''********************Data Cleaning End********************'''





'''++++++++++++++++++++++++REGRESSION TYPES++++++++++++++++++++++++'''





'''*******************Multiple Linear Regression*******************'''
MLR_regr = linear_model.LinearRegression()

'''Splitting dataset for training and testing.'''
MLR_X_train, MLR_X_test, MLR_y_train, MLR_y_test = train_test_split(
        feature_variables, target_variable, train_size=0.8, test_size=0.2)

MLR_regr.fit(MLR_X_train, MLR_y_train) # Finiding the fit equation
'''Predicting for target variable from test data.'''
MLR_y_pred = MLR_regr.predict(MLR_X_test) 

print(MLR_y_pred)
print('Coefficients: \n', MLR_regr.coef_) # The coefficients
print('Intercept: \n', MLR_regr.intercept_) # The intercept
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(MLR_y_test, MLR_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(MLR_y_test, MLR_y_pred))
for data in feature_variable_attributes:
    print("\n**********************"+data.upper()+"**********************")
    plt.xlabel(data.capitalize(), size=11, color="red")
    plt.ylabel("Value for money", size=11, color="red")
    plt.scatter(MLR_X_test[data], MLR_y_pred)
    plt.title("Japan Hostels - Multiple Linear Regression")
    plt.show()
'''*******************Multiple Linear Regression End*******************'''






'''*******************Ridge Regression*******************'''
RDG_regr = Ridge(fit_intercept=True, alpha=1.0, random_state=0, 
                 normalize=True)

'''Splitting dataset for training and testing.'''
RDG_X_train, RDG_X_test, RDG_y_train, RDG_y_test = train_test_split(
        feature_variables, target_variable, train_size=0.8, test_size=0.2)

RDG_regr.fit(RDG_X_train, RDG_y_train) # Finiding the fit equation
'''Predicting for target variable from test data.'''
RDG_y_pred = RDG_regr.predict(RDG_X_test) 

print(MLR_y_pred)
print('Coefficients: \n', RDG_regr.coef_) # The coefficients
print('Intercept: \n', RDG_regr.intercept_) # The intercept
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(RDG_y_test, RDG_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(RDG_y_test, RDG_y_pred))
for data in feature_variable_attributes:
    print("\n**********************"+data.upper()+"**********************")
    plt.xlabel(data.capitalize(), size=11, color="red")
    plt.ylabel("Value for money", size=11, color="red")
    plt.scatter(RDG_X_test[data], RDG_y_pred)
    plt.title("Japan Hostels - Ridge Regression")
    plt.show()
'''*******************Ridge Regression End*******************'''





'''*******************Lasso Regression*******************'''
Lasso_regr = linear_model.Lars(fit_intercept=True, fit_path=True,
   n_nonzero_coefs=1, normalize=True, positive=False, precompute='auto',
   verbose=False)

'''Splitting dataset for training and testing.'''
Lasso_X_train, Lasso_X_test, Lasso_y_train, Lasso_y_test = train_test_split(
        feature_variables, target_variable, train_size=0.8, test_size=0.2)

Lasso_regr.fit(Lasso_X_train, Lasso_y_train) # Finiding the fit equation

'''Predicting for target variable from test data.'''
Lasso_y_pred = Lasso_regr.predict(Lasso_X_test) 

print(Lasso_y_pred)
print('Coefficients: \n', Lasso_regr.coef_) # The coefficients
print('Intercept: \n', Lasso_regr.intercept_) # The intercept
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(Lasso_y_test, 
                                                      Lasso_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(Lasso_y_test, Lasso_y_pred))
for data in feature_variable_attributes:
    print("\n**********************"+data.upper()+"**********************")
    plt.xlabel(data.capitalize(), size=11, color="red")
    plt.ylabel("Value for money", size=11, color="red")
    plt.scatter(Lasso_X_test[data], Lasso_y_pred)
    plt.title("Japan Hostels - Lasso Regression")
    plt.show()
'''*******************Lasso Regression End*******************'''





'''*******************Elastic Regression*******************'''
ELST_regr = ElasticNet(alpha=1.0, fit_intercept=True, l1_ratio=0.5,
      max_iter=1000, normalize=False, positive=False, precompute=False,
      random_state=0, selection='cyclic', tol=0.0001, warm_start=False)

'''Splitting dataset for training and testing.'''
ELST_X_train, ELST_X_test, ELST_y_train, ELST_y_test = train_test_split(
        feature_variables, target_variable, train_size=0.8, test_size=0.2)

ELST_regr.fit(ELST_X_train, ELST_y_train) # Finiding the fit equation
'''Predicting for target variable from test data.'''
ELST_y_pred = ELST_regr.predict(Lasso_X_test) 

print(ELST_y_pred)
print('Coefficients: \n', ELST_regr.coef_) # The coefficients
print('Intercept: \n', ELST_regr.intercept_) # The intercept
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(ELST_y_test, 
                                                      ELST_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(ELST_y_test, ELST_y_pred))
for data in feature_variable_attributes:
    print("\n**********************"+data.upper()+"**********************")
    plt.xlabel(data.capitalize(), size=11, color="red")
    plt.ylabel("Value for money", size=11, color="red")
    plt.scatter(ELST_X_test[data], ELST_y_pred)
    plt.title("Japan Hostels - Elastic Regression")
    plt.show()
'''*******************Elastic Regression End*******************'''





#'''*******************Polynomial Regression*******************'''
#
#
#'''Splitting dataset for training and testing.'''
#POLY_X_train, POLY_X_test, POLY_y_train, POLY_y_test = train_test_split(
#        feature_variables, target_variable, train_size=0.8, test_size=0.2)
#print(POLY_X_train.shape)
#print(POLY_y_train.shape)
#
#'''Finiding the fit equation'''
#POLY_regr = PolynomialFeatures(degree=2)
#POLY_X_train_ = POLY_regr.fit_transform(POLY_X_train)
#POLY_y_train_ = POLY_regr.fit_transform(POLY_y_train)
#print(POLY_X_train_.shape)
#print(POLY_y_train_, len(POLY_y_train_[0]))
#print(POLY_y_train_.shape)
#
#'''Predicting for target variable from test data.'''
#Poly_LR = linear_model.LinearRegression()
#Poly_LR.fit(POLY_X_train_, POLY_y_train_)
#Poly_y_pred = Poly_LR.predict(POLY_y_test)
#
#print('Coefficients: \n', Poly_LR.coef_) # The coefficients
#print('Intercept: \n', Poly_LR.intercept_) # The intercept
## The mean squared error
#print("Mean squared error: %.2f" % mean_squared_error(POLY_y_test, 
#                                                      Poly_y_pred))
## Explained variance score: 1 is perfect prediction
#print('Variance score: %.2f' % r2_score(POLY_y_test, Poly_y_pred))
#for data in feature_variable_attributes:
#    print("\n**********************"+data.upper()+"**********************")
#    plt.xlabel(data.capitalize(), size=11, color="red")
#    plt.ylabel("Value for money", size=11, color="red")
#    plt.scatter(POLY_X_test[data], Poly_y_pred)
#    plt.title("Japan Hostels - Polynomail Regression")
#    plt.show()
#'''*******************Polynomial Regression End*******************'''







'''++++++++++++++++++++++++REGRESSION TYPES END++++++++++++++++++++++++'''



