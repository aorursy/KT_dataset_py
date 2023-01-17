import pandas as pd          

import numpy as np          # For mathematical calculations

import matplotlib.pyplot as plt  # For plotting graphs

from datetime import datetime    # To access datetime

from pandas import Series        # To work on series

%matplotlib inline

import warnings                   # To ignore the warnings

warnings.filterwarnings("ignore")
# Now let’s read the data

candies=pd.read_csv("../input/candy_production.csv")
candies_original=candies.copy()
candies.columns
candies.dtypes
candies.shape
candies.head()
candies.tail()
candies['observation_date'] = pd.to_datetime(candies.observation_date,format='%Y-%m-%d')  

candies_original['observation_date'] = pd.to_datetime(candies_original.observation_date,format='%Y-%m-%d')
#  let’s extract the year, month and day from the observation_date

for i in (candies,candies_original):

    i['year']=i.observation_date.dt.year 

    i['month']=i.observation_date.dt.month 

    i['day']=i.observation_date.dt.day
candies.head()
# let’s look at yearly production count.

plt.figure(figsize=(16,8))

candies.groupby('year')['IPG3113N'].mean().plot.bar()
# let’s look at monthly production count.

plt.figure(figsize=(16,8))

candies.groupby('month')['IPG3113N'].mean().plot.bar()
candies.index = candies['observation_date'] # indexing the Datetime to get the time period on the x-axis.

ts = candies['IPG3113N']

plt.figure(figsize=(16,8))

plt.plot(ts, label='% Candy Production')

plt.title('Candy Production')

plt.xlabel("Time(year)")

plt.ylabel("% Candy Production")

plt.legend(loc='best')
train=candies.ix[:'2011-10-01']

test=candies.ix['2011-11-01':]
train.head()
train.IPG3113N.plot(figsize=(15,8), title= 'Candy Production', fontsize=14, label='train')

test.IPG3113N.plot(figsize=(15,8), title= 'Candy Production', fontsize=14, label='test')

plt.xlabel("observation_date")

plt.ylabel("production count")

plt.legend(loc='best')

plt.show()
# predictions using naive approach for the validation set.

dd= np.asarray(train['IPG3113N'])

y_hat = test.copy()

y_hat['naive'] = dd[len(dd)-1]

plt.figure(figsize=(12,8))

plt.plot(train.index, train['IPG3113N'], label='Train')

plt.plot(test.index,test['IPG3113N'], label='Test')

plt.plot(y_hat.index,y_hat['naive'], label='Naive Forecast')

plt.legend(loc='best')

plt.title("Naive Forecast")

plt.show()
# RMSE(Root Mean Square Error) to check the accuracy of our model on validation data set.

from sklearn.metrics import mean_squared_error

from math import sqrt

rms = sqrt(mean_squared_error(test['IPG3113N'], y_hat.naive))

print(rms)
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

y_hat_ex = test.copy()

fit2 = SimpleExpSmoothing(np.asarray(train['IPG3113N'])).fit(smoothing_level=0.6,optimized=False)

y_hat_ex['SES'] = fit2.forecast(len(test))

plt.figure(figsize=(16,8))

plt.plot(train['IPG3113N'], label='Train')

plt.plot(test['IPG3113N'], label='Test')

plt.plot(y_hat_ex['SES'], label='SES')

plt.legend(loc='best')

plt.show()
rms = sqrt(mean_squared_error(test['IPG3113N'], y_hat_ex['SES']))

print(rms)
from statsmodels.tsa.statespace.sarimax import SARIMAX



y_hat_avg = test.copy()

fit1 = SARIMAX(train['IPG3113N'], order=(2, 1, 4),seasonal_order=(0,1,1,7),enforce_stationarity=False,enforce_invertibility=False).fit()

y_hat_ex['SARIMA'] = fit1.predict(start="2011-11-01", end="2017-08-01", dynamic=True)

plt.figure(figsize=(16,8))

plt.plot( train['IPG3113N'], label='Train')

plt.plot(test['IPG3113N'], label='Test')

plt.plot(y_hat_ex['SARIMA'], label='SARIMA')

plt.legend(loc='best')

plt.show()
# Let’s check the rmse value for the validation part.



rms = sqrt(mean_squared_error(test['IPG3113N'], y_hat_ex.SARIMA))

print(rms)
candies_original['IPG3113N'] = pd.to_numeric(candies_original['IPG3113N'], errors='coerce')
candies_original['observation_date'] = pd.to_numeric(candies_original['observation_date'], errors='coerce')
# Machine Learning

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split



X = candies_original.iloc[:, :1]

y = candies_original.iloc[:, :2]



# Create training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)



# Create the regressor: reg_all

reg_all = LinearRegression()



# Fit the regressor to the training data

reg_all.fit(X_train, y_train)



# Predict on the test data: y_pred

y_pred = reg_all.predict(X_test)



# Compute and print R^2 and RMSE

print("R^2: {}".format(reg_all.score(X_test, y_test)))

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Root Mean Squared Error: {}".format(rmse))
