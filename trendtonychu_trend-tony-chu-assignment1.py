import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

t0 = pd.read_csv('../input/train.csv')

# look for the columns with values > 0.5, and found these:
t0.corr().nlargest(20, 'SalePrice')['SalePrice']
# ... and found these:
# OverallQual
# GrLivArea
# GarageCars
# GarageArea
# TotalBsmtSF
# 1stFlrSF
# FullBath
# TotRmsAbvGrd
# YearBuilt
# YearRemodAdd
train = t0[['SalePrice',
            'OverallQual',
            'GrLivArea',
            'GarageCars',
            'GarageArea',
            'TotalBsmtSF',
            '1stFlrSF',
            'FullBath',
            'TotRmsAbvGrd',
            'YearBuilt',
            'YearRemodAdd']].copy()
train.describe()
# some of them are quite large, apply log() on them
# adding 1 because some of them have 0 values
train['SalePrice'] = np.log(train['SalePrice'])
train['GrLivArea'] = np.log(train['GrLivArea'] + 1)
train['GarageArea'] = np.log(train['GarageArea'] + 1)
train['TotalBsmtSF'] = np.log(train['TotalBsmtSF'] + 1)
train['1stFlrSF'] = np.log(train['1stFlrSF'] + 1)
train['YearBuilt'] = np.log(train['YearBuilt'] + 1)
train['YearRemodAdd'] = np.log(train['YearRemodAdd'] + 1)
train.describe()
# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
train_x = train.iloc[:, [1,2,3,4,5,6,7,8,9,10]].values
train_y = train.iloc[:, [0]].values
linear = LinearRegression()
linear.fit(train_x, train_y)
print(linear.intercept_)
print(linear.coef_)
prediction = linear.predict(train_x)
mean_squared_error(prediction, train_y)
# try the test data
t0 = pd.read_csv('../input/test.csv')
t0.describe()
# calling fillna(0) because some values are "NA" which will cause problems
test = t0[['Id',
           'OverallQual',
           'GrLivArea',
           'GarageCars',
           'GarageArea',
           'TotalBsmtSF',
           '1stFlrSF',
           'FullBath',
           'TotRmsAbvGrd',
           'YearBuilt',
           'YearRemodAdd']].copy().fillna(0)
t0.describe()
test['GrLivArea'] = np.log(test['GrLivArea'] + 1)
test['GarageArea'] = np.log(test['GarageArea'] + 1)
test['TotalBsmtSF'] = np.log(test['TotalBsmtSF'] + 1)
test['1stFlrSF'] = np.log(test['1stFlrSF'] + 1)
test['YearBuilt'] = np.log(test['YearBuilt'] + 1)
test['YearRemodAdd'] = np.log(test['YearRemodAdd'] + 1)
test.describe()
test_x = test.iloc[:, [1,2,3,4,5,6,7,8,9,10]].values
sales_price_log = linear.predict(test_x)

output = pd.concat([test['Id'], pd.DataFrame(np.exp(sales_price_log))], axis=1)
output.columns = ['Id', 'SalesPrice_Prediction']
output.describe()
output.to_csv('assignment1_linear.csv', index=False)