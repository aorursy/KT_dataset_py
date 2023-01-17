# load library

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sb
# load dataset

cnames = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style',

                'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight',

                'engine-type', 'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio',

                'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']

data = pd.read_csv('/kaggle/input/automobile-dataset/Automobile_data.csv')

data = data.replace('?', np.nan)
data.shape
data.info()
data.head()
data.drop(columns= 'symboling', inplace= True)

data.drop(columns= 'normalized-losses', inplace= True)

data.drop(columns= 'engine-location', inplace= True)

data.drop(columns= 'wheel-base', inplace= True)

data.drop(columns= 'engine-type', inplace= True)

data.drop(columns= 'bore', inplace= True)

data.drop(columns= 'stroke', inplace= True)

data.drop(columns= 'highway-mpg', inplace= True)

data.drop(columns= 'compression-ratio', inplace= True)

data.drop(columns= 'width', inplace= True)

data.drop(columns= 'height', inplace= True)

# format horsepower

data['horsepower'] = pd.to_numeric(data['horsepower'])

# format engine-size

data['engine-size'] = pd.to_numeric(data['engine-size'])

# format num_of_cylinders

num_of_cylinders_code = {'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'eight': 8, 'twelve': 12}

data['num-of-cylinders'] = data['num-of-cylinders'].map(num_of_cylinders_code)

# format num_of_doors

num_of_doors_code = {'two': 2, 'four': 4}

data['num-of-doors'] = data['num-of-doors'].map(num_of_doors_code)

# format peak-rpm

data['peak-rpm'] = pd.to_numeric(data['peak-rpm'])

# format price

data['price'] = pd.to_numeric(data['price'])
# View data type after formatting

data.info()
import scipy

import sys

for i in ['engine-size', 'horsepower', 'city-mpg', 'price']:

    upper_outlayer = data[i] > (np.percentile(data[i], 75) + 1.5* scipy.stats.iqr(data[i]))

    lower_outlayer = data[i] < (np.percentile(data[i], 25) - 1.5* scipy.stats.iqr(data[i]))

    outlayer_all = lower_outlayer | upper_outlayer

    data[i].loc[outlayer_all] = data[i].mean()
# Detect variable inclue missing value

data.isna().sum(0)
# Solve missing values

data['num-of-doors'].loc[data['num-of-doors'].isna()] = data['num-of-doors'].mode()[0]

for i in ['horsepower', 'horsepower', 'peak-rpm', 'price']:

    data[i].loc[data[i].isna()] = data[i].mean()
# recode aspiration

aspiration_code = {'std': 0, 'turbo': 1}

data['aspiration'] = data['aspiration'].map(aspiration_code)



# recode fuel-type

fuel_type_code = {'diesel': 0, 'gas': 1}

data['fuel-type'] = data['fuel-type'].map(fuel_type_code)



# Onehot endcoding

data = pd.get_dummies(data= data, prefix= ['make', 'body-style', 'drive-wheels', 'fuel-system'])
data.head()
y = data['price'].values

x = data.drop(columns = ['price'])

# MinMaxScaler

from sklearn.preprocessing import MinMaxScaler

x= MinMaxScaler().fit(x).transform(x)
# Split dataset to train and test datasets

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=4)
# Building linear model

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()  

regressor.fit(x_train, y_train)
# Predict on test set

y_pred = regressor.predict(x_test)
r = scipy.stats.pearsonr(y_test, y_pred)

pearson_r = str(round(r[0],2))

pearson_p = str(r[1])



plt.figure(figsize= (10,4))

ax1 = plt.subplot(121)

sb.scatterplot(y_test, y_pred)

plt.text(5000, 35000, 'pearson R: '+ pearson_r + '\np: ' + pearson_p, va = 'top')



ax2 = plt.subplot(122)

sb.kdeplot(y_test, shade=True, color="r", legend=True, label = 'Test')

sb.kdeplot(y_pred, shade=True, color="b", legend=True, label = 'Predict')

plt.legend()
# Mean squared error

from sklearn.metrics import mean_squared_error

MSE = mean_squared_error(y_test,y_pred)

print('Mean squared error:', MSE)



# Mean absolute error

from sklearn.metrics import mean_absolute_error

MAE = mean_absolute_error(y_test,y_pred)

print('Mean absolute error:', MAE)



# R2 score

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)

print('R2 score:', r2)
from sklearn.linear_model import Ridge

ridge = Ridge(alpha = 0.05, normalize = True)

ridge.fit(x_train, y_train)
y_pred = ridge.predict(x_test)
r = scipy.stats.pearsonr(y_test, y_pred)

pearson_r = str(round(r[0],2))

pearson_p = str(r[1])



plt.figure(figsize= (10,4))

ax1 = plt.subplot(121)

sb.scatterplot(y_test, y_pred)

plt.text(5000, 35000, 'pearson R: '+ pearson_r + '\np: ' + pearson_p, va = 'top')



ax2 = plt.subplot(122)

sb.kdeplot(y_test, shade=True, color="r", legend=True, label = 'Test')

sb.kdeplot(y_pred, shade=True, color="b", legend=True, label = 'Predict')

plt.legend()
# Mean squared error

from sklearn.metrics import mean_squared_error

MSE = mean_squared_error(y_test,y_pred)

print('Mean squared error:', MSE)



# Mean absolute error

from sklearn.metrics import mean_absolute_error

MAE = mean_absolute_error(y_test,y_pred)

print('Mean absolute error:', MAE)



# R2 score

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)

print('R2 score:', r2)
from sklearn.linear_model import Lasso

lasso_model = Lasso(alpha = 0.05, normalize = True)

lasso_model.fit(x_train, y_train)
y_pred = lasso_model.predict(x_test)
r = scipy.stats.pearsonr(y_test, y_pred)

pearson_r = str(round(r[0],2))

pearson_p = str(r[1])



plt.figure(figsize= (10,4))

ax1 = plt.subplot(121)

sb.scatterplot(y_test, y_pred)

plt.text(5000, 35000, 'pearson R: '+ pearson_r + '\np: ' + pearson_p, va = 'top')



ax2 = plt.subplot(122)

sb.kdeplot(y_test, shade=True, color="r", legend=True, label = 'Test')

sb.kdeplot(y_pred, shade=True, color="b", legend=True, label = 'Predict')

plt.legend()
# Mean squared error

from sklearn.metrics import mean_squared_error

MSE = mean_squared_error(y_test,y_pred)

print('Mean squared error:', MSE)



# Mean absolute error

from sklearn.metrics import mean_absolute_error

MAE = mean_absolute_error(y_test,y_pred)

print('Mean absolute error:', MAE)



# R2 score

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)

print('R2 score:', r2)
from sklearn.linear_model import ElasticNet

enet_model = ElasticNet(alpha=0.01, l1_ratio=0.5, normalize=False)

enet_model.fit(x_train, y_train)
y_pred = enet_model.predict(x_test)
r = scipy.stats.pearsonr(y_test, y_pred)

pearson_r = str(round(r[0],2))

pearson_p = str(r[1])



plt.figure(figsize= (10,4))

ax1 = plt.subplot(121)

sb.scatterplot(y_test, y_pred)

plt.text(5000, 35000, 'pearson R: '+ pearson_r + '\np: ' + pearson_p, va = 'top')



ax2 = plt.subplot(122)

sb.kdeplot(y_test, shade=True, color="r", legend=True, label = 'Test')

sb.kdeplot(y_pred, shade=True, color="b", legend=True, label = 'Predict')

plt.legend()
# Mean squared error

from sklearn.metrics import mean_squared_error

MSE = mean_squared_error(y_test,y_pred)

print('Mean squared error:', MSE)



# Mean absolute error

from sklearn.metrics import mean_absolute_error

MAE = mean_absolute_error(y_test,y_pred)

print('Mean absolute error:', MAE)



# R2 score

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)

print('R2 score:', r2)
from sklearn.ensemble import RandomForestRegressor

RandomForest = RandomForestRegressor()

RandomForest.fit(x_train, y_train)
y_pred = RandomForest.predict(x_test)
r = scipy.stats.pearsonr(y_test, y_pred)

pearson_r = str(round(r[0],2))

pearson_p = str(r[1])



plt.figure(figsize= (10,4))

ax1 = plt.subplot(121)

sb.scatterplot(y_test, y_pred)

plt.text(5000, 35000, 'pearson R: '+ pearson_r + '\np: ' + pearson_p, va = 'top')



ax2 = plt.subplot(122)

sb.kdeplot(y_test, shade=True, color="r", legend=True, label = 'Test')

sb.kdeplot(y_pred, shade=True, color="b", legend=True, label = 'Predict')

plt.legend()
# Mean squared error

from sklearn.metrics import mean_squared_error

MSE = mean_squared_error(y_test,y_pred)

print('Mean squared error:', MSE)



# Mean absolute error

from sklearn.metrics import mean_absolute_error

MAE = mean_absolute_error(y_test,y_pred)

print('Mean absolute error:', MAE)



# R2 score

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)

print('R2 score:', r2)