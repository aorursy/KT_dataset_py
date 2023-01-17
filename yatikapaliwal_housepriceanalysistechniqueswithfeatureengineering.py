import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OrdinalEncoder
from scipy import stats
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.preprocessing import OrdinalEncoder
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train.head()
test.head()
print('the shape of train is: ',train.shape)
print('the shape of test is: ',test.shape)
train.isnull().sum().sort_values(ascending = False)
train.info()
train = train.drop('Id', axis=1)
train.columns
plt.figure(figsize=(10,7))
sns.distplot(train['SalePrice'])
num_cols = train.select_dtypes(include=[np.int64]).columns
num_cols.append(train.select_dtypes(include=[np.float64]).columns)
cat_cols = train.select_dtypes(include=[np.object]).columns
num_cols
#to check the distribution of numeric cols
plt.figure(figsize=(30,50))
for i in range(len(num_cols)):
  plt.subplot(15,3,i+1)
  sns.distplot(train[num_cols[i]], kde= False)
  plt.show
plt.figure(figsize=(30,50))
for i in range(len(num_cols)):
  plt.subplot(15,3,i+1)
  sns.boxplot(train[num_cols[i]])
  plt.show
plt.figure(figsize=(30,50))
for i in range(len(cat_cols)):
  plt.subplot(17,3,i+1)
  sns.barplot(train[cat_cols[i]], train['SalePrice'])
  plt.show
plt.figure(figsize=(30,50))
for i in range(len(num_cols)):
  plt.subplot(15,3,i+1)
  sns.scatterplot(train[num_cols[i]], train['SalePrice'])
  plt.show
plt.figure(figsize=(20,15))
sns.heatmap(train[num_cols].corr(), annot= True, cbar=True)
plt.figure(figsize=(20,7))
sns.lineplot(x = 'YearBuilt', y= 'SalePrice', data= train)
plt.xticks(rotation = 90, ha = 'right');
plt.figure(figsize=(10,6))
sns.lineplot(x = 'YrSold', y= 'SalePrice', data= train)
plt.figure(figsize=(10,6))
sns.lineplot(x = 'MoSold', y= 'SalePrice', data= train)
#missing data
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing = pd.concat([percent], axis=1, keys=['Percent'])
missing.head(20)
#we are reomoving top 5 missing data columns since there are lot of missing values(i.e more than 10-15% missing values)
train = train.drop(columns =['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage'], axis =1)
test = test.drop(columns =['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage'], axis =1)
#Lets explore other variable if they are important ones
train[['GarageCond', 'GarageQual', 'GarageFinish', 'GarageType', 'GarageYrBlt', 'BsmtFinType1', 'BsmtExposure', 'BsmtQual','BsmtCond', 'BsmtFinType1', 'MasVnrArea', 'MasVnrType', 'Electrical']].isnull().sum()
train =train.drop(columns =['GarageQual', 'GarageFinish', 'GarageType', 'GarageYrBlt', 'BsmtFinType1', 'BsmtExposure', 'BsmtQual','BsmtFinType1', 'MasVnrType'], axis =1)
train =train.dropna(axis =0)
train.info()
train.select_dtypes(np.object).head(2)
cat_cols = train.select_dtypes(np.object).columns
cat_cols
for i in range(len(cat_cols)):
  train[cat_cols[i]] = train[cat_cols[i]].astype('category')
  train[cat_cols[i]] = train[cat_cols[i]].cat.codes
for i in range(len(cat_cols)):
  test[cat_cols[i]] = test[cat_cols[i]].astype('category')
  test[cat_cols[i]] = test[cat_cols[i]].cat.codes
plt.figure(figsize=(40,15))
sns.heatmap(train.corr(), annot= True, cbar=True)
train = train.drop(columns = ['LotArea', 'YearBuilt', 'BsmtFinSF1', 'BsmtUnfSF', '2ndFlrSF', 'OpenPorchSF', 'WoodDeckSF', 'PoolArea', 'YrSold', 'MoSold'], axis =1)

test = test.drop(columns = ['LotArea', 'YearBuilt', 'BsmtFinSF1', 'BsmtUnfSF', '2ndFlrSF', 'OpenPorchSF', 'WoodDeckSF', 'PoolArea', 'YrSold', 'MoSold'], axis =1)
#normality test on target variable
plt.figure(figsize=(12,5))
import pylab
plt.subplot(1,2,1)
sns.distplot(train['SalePrice'])
plt.subplot(1,2,2)
#calculates a best-fit line for the data and plots the results using Matplotlib or a given plot function.
res = stats.probplot(train['SalePrice'], plot=plt)
pylab.show()
# Shapiro-Wilk Test
from scipy.stats import shapiro
data = train['SalePrice']
# normality test
stat, p = shapiro(data)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Sample looks Gaussian')
else:
	print('Sample does not look Gaussian')
train['SalePrice'] = np.log(train['SalePrice'])
#normality test on target variable
plt.figure(figsize=(12,5))
import pylab
plt.subplot(1,2,1)
sns.distplot(train['SalePrice'])
plt.subplot(1,2,2)
#calculates a best-fit line for the data and plots the results using Matplotlib or a given plot function.
res = stats.probplot(train['SalePrice'], plot=plt)
pylab.show()
train.head()
plt.figure(figsize=(12,5))
import pylab
plt.subplot(1,2,1)
sns.distplot(train['TotalBsmtSF'])
plt.subplot(1,2,2)
#calculates a best-fit line for the data and plots the results using Matplotlib or a given plot function.
res = stats.probplot(train['TotalBsmtSF'], plot=plt)
pylab.show()
train['TotalBsmtSF'] = np.log(train['TotalBsmtSF'])
#normality test on target variable
plt.figure(figsize=(12,5))
import pylab
plt.subplot(1,2,1)
sns.distplot(train['TotalBsmtSF'])
plt.subplot(1,2,2)
#calculates a best-fit line for the data and plots the results using Matplotlib or a given plot function.
res = stats.probplot(train['TotalBsmtSF'], plot=plt)
pylab.show()
plt.figure(figsize=(12,5))
import pylab
plt.subplot(1,2,1)
sns.distplot(train['GrLivArea'])
plt.subplot(1,2,2)
#calculates a best-fit line for the data and plots the results using Matplotlib or a given plot function.
res = stats.probplot(train['GrLivArea'], plot=plt)
pylab.show()
train['GrLivArea'] = np.log(train['GrLivArea'])
plt.figure(figsize=(12,5))
import pylab
plt.subplot(1,2,1)
sns.distplot(train['1stFlrSF'])
plt.subplot(1,2,2)
#calculates a best-fit line for the data and plots the results using Matplotlib or a given plot function.
res = stats.probplot(train['1stFlrSF'], plot=plt)
pylab.show()
train['1stFlrSF'] = np.log(train['1stFlrSF'])
plt.figure(figsize=(12,5))
import pylab
plt.subplot(1,2,1)
sns.distplot(train['GarageArea'])
plt.subplot(1,2,2)
#calculates a best-fit line for the data and plots the results using Matplotlib or a given plot function.
res = stats.probplot(train['GarageArea'], plot=plt)
pylab.show()
train['GarageArea'] = np.log(train['GarageArea'])
train.shape
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
X = train.drop('SalePrice', axis =1)
y = train['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state =100, test_size =0.2)
model_Rf = RandomForestRegressor()
model_Rf.fit(X_train, y_train)
y_pred = model_Rf.predict(X_test)

rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
SSE = np.sum((y_pred-y_test)**2)
SST = np.sum((y_test-np.mean(y_train))**2)
r2_test = 1 - SSE/SST
print("Test RMSE : ", rmse_test)
print("Test SSE : ", SSE)
print("Test SST : ", SST)
print("Test R2 : ", r2_test)
model_GB = GradientBoostingRegressor()
model_GB.fit(X_train, y_train)
y_pred = model_GB.predict(X_test)

rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
SSE = np.sum((y_pred-y_test)**2)
SST = np.sum((y_test-np.mean(y_train))**2)
r2_test = 1 - SSE/SST
print("Test RMSE : ", rmse_test)
print("Test SSE : ", SSE)
print("Test SST : ", SST)
print("Test R2 : ", r2_test)