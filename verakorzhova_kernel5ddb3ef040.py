# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataInfo = open('/kaggle/input/house-prices-advanced-regression-techniques/data_description.txt', 'r')
for line in dataInfo:
    print(line)
train_df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
train_df
train_df.info()
train_df.describe()
import matplotlib.pyplot as plt
plt.scatter(train_df.SalePrice, train_df.LotArea)
plt.title("Price vs Lot Area")
plt.scatter(train_df.SalePrice, train_df.YearBuilt)
plt.title("Price vs YearBuilt")
plt.scatter(train_df.BedroomAbvGr, train_df.SalePrice)
plt.title("Price vs BedroomAbvGr")
plt.xlabel("Bedrooms")
plt.ylabel("Price")
plt.scatter(train_df.SalePrice, train_df.GrLivArea)
plt.title("Price vs Liv Area")
plt.scatter(train_df.GarageCars, train_df.SalePrice)
plt.title("Price vs Garage for cars")

plt.scatter(train_df.FullBath, train_df.SalePrice)
plt.title("Price vs Full bathrooms above grade")

plt.scatter(train_df.OverallQual, train_df.SalePrice)
plt.title("Price vs OverallQual")
total = train_df.isnull().sum().sort_values(ascending=False)
percent = (train_df.isnull().sum()/train_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
train_df = train_df.drop((missing_data[missing_data['Total'] > 1]).index,1)
train_df = train_df.drop(train_df.loc[train_df['Electrical'].isnull()].index)
train_df.isnull().sum().max()
X = pd.get_dummies(train_df)
Y = train_df['SalePrice'].values

X = X.drop(['SalePrice'], axis = 1)
X.info()


# useful_cols = ['LotArea', 'GrLivArea', 'OverallQual', 'GarageCars', 'TotalBsmtSF', 'BedroomAbvGr', 'FullBath']


# X = pd.get_dummies(train_df[useful_cols])
# Y = train_df['SalePrice'].values
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Ridge
params_ridge ={
        'alpha':[0.25,0.5,1],
        'solver':['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
        }
ridge = Ridge()
ridge_random = RandomizedSearchCV(estimator = ridge, param_distributions = params_ridge,
                               n_iter=50, cv=5, n_jobs=-1,random_state=42, verbose=2)
ridge_random.fit(X_train, Y_train)
print(ridge_random.best_params_)
print(ridge_random.best_score_)

ridge_grid = GridSearchCV(estimator = ridge, param_grid = params_ridge, cv = 5, n_jobs = -1, verbose = 2)
ridge_grid.fit(X_train, Y_train)
print(ridge_grid.best_params_)
print(ridge_grid.best_score_)
import seaborn as sns
sns.set_context("notebook", font_scale=1.1)
sns.set_style("ticks")
sns.lmplot('GrLivArea','SalePrice', data=train_df)
plt.ylabel('Response')
plt.xlabel('Explanatory')
from sklearn import linear_model 
from sklearn.linear_model import LinearRegression

linear = linear_model.LinearRegression()
linear.fit(X_train, Y_train)
linear.score(X_train, Y_train)
# print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)
print('R² Value: \n', linear.score(X_train, Y_train))
predicted = linear.predict(X_test)
print(predicted)