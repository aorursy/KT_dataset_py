# Importing required libraries.

import pandas as pd

import numpy as np

import seaborn as sns #visualisation

import matplotlib.pyplot as plt #visualisation

%matplotlib inline 

sns.set(color_codes=True)

import os



from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

import statsmodels.api as sm

from statsmodels.tools.eval_measures import mse, rmse

pd.options.display.float_format = '{:.5f}'.format

import warnings

import math

import scipy.stats as stats

import scipy

from sklearn.preprocessing import scale

# warnings.filterwarnings('ignore')
#Import train dataset

# os.chdir('/kaggle/input')

df = pd.read_csv('../input/train.csv')

df.head(5)
# No duplicates found

duplicate_rows_df = df[df.duplicated()]

print('Total duplicate rows:', duplicate_rows_df.shape)

print('\nShape of Dataframe:',df.shape,'\n\nDataTypes:\n',df.dtypes)
df.describe(include='all')
df.info()
new_df = df.rename(columns={'loc.details':'details','location.Code':'branch_code','deposit_amount_2011':'2011','deposit_amount_2012':'2012','deposit_amount_2013':'2013','deposit_amount_2014':'2014','deposit_amount_2015':'2015','deposit_amount_2016':'2016','deposit_amount_2017':'Target'})

new_df.head()
# Finding the relations between the variables.

plt.figure(figsize=(20,10))

map= new_df.corr()

sns.heatmap(map,cmap='BrBG',annot=True)
grid = sns.FacetGrid(new_df, size=2.2, aspect=5.6)

grid.map(sns.pointplot, 'state', 'Target', palette='deep')

grid.add_legend()
train = new_df.drop(columns ={'branch_code','id','headquarter','date_of_establishment','location','details'}, axis = 1)

train.head()
# Finding the null values

print(train.isnull().sum())

print(train.shape)

# train = train.dropna() 
# Remove Outliers

Q1 = train.quantile(0.25)

Q3 = train.quantile(0.75)

IQR = Q3 - Q1

IQR
train = train[~((train < (Q1-1.5 * IQR)) |(train > (Q3 + 1.5 * IQR))).any(axis=1)]

train.shape
# Filling Missing Values with mean

train['2011'].fillna(train['2011'].mean(), inplace=True)

train['2012'].fillna(train['2012'].mean(), inplace=True)

train['2013'].fillna(train['2013'].mean(), inplace=True)

train['2014'].fillna(train['2014'].mean(), inplace=True)

train['2015'].fillna(train['2015'].mean(), inplace=True)

train = train.dropna() 
# Adding Dummy Variables

# generate binary values using get_dummies

train = pd.get_dummies(train, columns=["state"], prefix=["state"] )

train=train[['2011','2012','2013','2014','2015','2016','state_OH'

,'state_NY'

,'state_CT'

,'state_NJ'

,'state_TX'

,'state_KY'

# ,'state_WV'

,'state_IL'

,'state_LA'

,'state_FL'

,'state_AZ'

,'state_UT'

,'state_MI'

,'state_WI','Target']]

train.head()
# Finding the relations between the variables.

plt.figure(figsize=(40,10))

map= train.corr()

sns.heatmap(map,cmap='BrBG',annot=True)
Y=train['Target']

X = train.drop(columns ={'Target'}, axis = 1)



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 465)



print('Training Data Count: {}'.format(X_train.shape[0]))

print('Testing Data Count: {}'.format(X_test.shape[0]))
X_train = sm.add_constant(X_train)

results = sm.OLS(y_train, X_train).fit()

results.summary()
X_test = sm.add_constant(X_test)



y_preds = results.predict(X_test)



print("Mean Absolute Error (MAE)         : {}".format(mean_absolute_error(y_test, y_preds)))

print("Mean Squared Error (MSE) : {}".format(mse(y_test, y_preds)))

print("Root Mean Squared Error (RMSE) : {}".format(rmse(y_test, y_preds)))

print("Root Mean Squared Error (RMSE) : {}".format(rmse(y_test, y_preds)))

print("Mean Absolute Perc. Error (MAPE) : {}".format(np.mean(np.abs((y_test - y_preds) / y_test)) * 100))
X2 = train[['2016','2015','2014','2013','2011']]

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, Y, test_size = 0.3, random_state = 465)



print('Training Data Count:', X2_train.shape)

print('Testing Data Count::', X2_test.shape)





X2_train = sm.add_constant(X2_train)



results2 = sm.OLS(y2_train, X2_train).fit()



results2.summary()
X2_test = sm.add_constant(X2_test)



y2_preds = results2.predict(X2_test)



print("Mean Absolute Error (MAE)         : {}".format(mean_absolute_error(y2_test, y2_preds)))

print("Mean Squared Error (MSE)          : {}".format(mse(y2_test, y2_preds)))

print("Root Mean Squared Error (RMSE)    : {}".format(rmse(y2_test, y2_preds)))

print("Root Mean Squared Error (RMSE)    : {}".format(rmse(y2_test, y2_preds)))

print("Mean Absolute Perc. Error (MAPE)  : {}".format(np.mean(np.abs((y2_test - y2_preds) / y2_test)) * 100))
test = pd.read_csv('../input/test.csv')

test.head(5)
# data = [[32079.00000	,35971.50000	,37237.50000	,40362.00000	,46021.50000]]

test1 = pd.get_dummies(test, columns=["state"], prefix=["state"] )

test1 = test1.drop(columns ={'location.Code','id','headquarter','date_of_establishment','location','loc.details','state_WV'}, axis = 1)



# test1

test1 = sm.add_constant(test1)

test['deposit_amount_2017'] = results.predict(test1)

# test = results2.predict(data)

# test.to_csv('Output3.csv')
test