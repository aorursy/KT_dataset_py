# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# LinearRegression is a machine learning library for linear regression 

! pip install yfinance



import yfinance as yf

from sklearn.linear_model import LinearRegression 

# pandas and numpy are used for data manipulation 



import pandas as pd 



import numpy as np 

# matplotlib and seaborn are used for plotting graphs 



import matplotlib.pyplot as plt 



import seaborn 

# fix_yahoo_finance is used to fetch data 
# Read data 



df = yf.download('GLD','2010-01-01','2019-12-31')

# Only keep close columns 



df1=df[['Close']] 

# Drop rows with missing values 





# Plot the closing price of GLD 





df1.Close.plot(figsize=(10,5)) 



plt.ylabel("Gold ETF Prices")



plt.show()
from statsmodels.tsa.stattools import adfuller

from numpy import log

result = adfuller(df1)

print('ADF Statistic: %f' % result[0])

print('p-value: %f' % result[1])



#The null hypothesis of the ADF test is that the time series is non-stationary. 

#So, if the p-value of the test is less than the significance level (0.05) 

#then you reject the null hypothesis and infer that the time series is indeed stationary
#Since P-value is greater than the significance level, 

#let’s difference the series and see how the autocorrelation plot looks like.

import numpy as np, pandas as pd

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import matplotlib.pyplot as plt

plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})
!ls ../input/algo-supplementary/algo.csv
import numpy as np, pandas as pd

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import matplotlib.pyplot as plt

plt.rcParams.update({'figure.figsize':(12,8), 'figure.dpi':120})
# Original Series

import pandas as pd

df2= pd.read_csv('../input/algo-supplementary/algo.csv')



fig, axes = plt.subplots(3, 2, sharex=True)

axes[0, 0].plot(df2); axes[0, 0].set_title('Original Series')

plot_acf(df2, ax=axes[0,1])



# 1st Differencing

axes[1, 0].plot(df2.diff()); axes[1, 0].set_title('1st Order Differencing')

plot_acf(df2.diff().dropna(), ax=axes[1, 1])



# 2nd Differencing

axes[2, 0].plot(df2.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')

plot_acf(df2.diff().diff().dropna(), ax=axes[2, 1])

         

plt.show()
# 3rd Series

import pandas as pd

df2= pd.read_csv('../input/algo-supplementary/algo.csv')



fig, axes = plt.subplots(3, 2, sharex=True)

axes[0, 0].plot(df2.diff().diff().diff()); axes[0, 0].set_title('3rd Order Differencing')

plot_acf(df2.diff().diff().diff().dropna(), ax=axes[0, 1])





# 4th Differencing

axes[1, 0].plot(df2.diff().diff().diff().diff()); axes[1, 0].set_title('4th Order Differencing')

plot_acf(df2.diff().diff().diff().diff().dropna(), ax=axes[1, 1])



# 5th Differencing

axes[2, 0].plot(df2.diff().diff().diff().diff().diff()); axes[2, 0].set_title('5th Order Differencing')

plot_acf(df2.diff().diff().diff().diff().diff().dropna(), ax=axes[2, 1])

         

plt.show()
# PACF plot of 1st differenced series

plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})



fig, axes = plt.subplots(1, 2, sharex=True)

axes[0].plot(df2.diff()); axes[0].set_title('1st Differencing')

axes[1].set(ylim=(0,1.2))

plot_pacf(df2.diff().dropna(), ax=axes[1])

from statsmodels.tsa.arima_model import ARIMA



# 1,1,2 ARIMA Model

model = ARIMA(df2, order=(1,1,2))

model_fit = model.fit(disp=0)

print(model_fit.summary())
# 1,1,1 ARIMA Model

model = ARIMA(df2, order=(1,1,1))

model_fit = model.fit(disp=0)

print(model_fit.summary())
# Plot residual errors

residuals = pd.DataFrame(model_fit.resid)

fig, ax = plt.subplots(1,2)

residuals.plot(title="Residuals", ax=ax[0])

residuals.plot(kind='kde', title='Density', ax=ax[1])

plt.show()
# Actual vs Fitted

model_fit.plot_predict(dynamic=False)

plt.plot(label='actual')

plt.show()
train_ratio = 0.8



data_size = len(df2)

print (data_size)
from statsmodels.tsa.stattools import acf



train= round(train_ratio * data_size)

print (train)

# Create Training and Test

train= df2[:2011]

test = df2[2011:]
# Build Model

# model = ARIMA(train, order=(3,2,1))  

model = ARIMA(train, order=(1, 1, 1))  

fitted = model.fit(disp=1)  



# Forecast

fc, se, conf = fitted.forecast(503, alpha=0.05)  # 95% conf



# Make as pandas series

fc_series = pd.Series(fc, index=test.index)

lower_series = pd.Series(conf[:, 0], index=test.index)

upper_series = pd.Series(conf[:, 1], index=test.index)



# Plot

plt.figure(figsize=(12,5), dpi=100)

plt.plot(train, label='training')

plt.plot(test, label='actual')

plt.plot(fc_series, label='forecast')

plt.fill_between(lower_series.index, lower_series, upper_series, 

                 color='k', alpha=.15)

plt.title('Forecast vs Actuals')

plt.legend(loc='upper left', fontsize=8)

plt.show
# Build Model

model = ARIMA(train, order=(3, 2, 1))  

fitted = model.fit(disp=-1)  

print(fitted.summary())



# Forecast

fc, se, conf = fitted.forecast(503, alpha=0.05)  # 95% conf



# Make as pandas series

fc_series = pd.Series(fc, index=test.index)

lower_series = pd.Series(conf[:, 0], index=test.index)

upper_series = pd.Series(conf[:, 1], index=test.index)



# Plot

plt.figure(figsize=(12,5), dpi=100)

plt.plot(train, label='training')

plt.plot(test, label='actual')

plt.plot(fc_series, label='forecast')

plt.fill_between(lower_series.index, lower_series, upper_series, 

                 color='k', alpha=.15)

plt.title('Forecast vs Actuals')

plt.legend(loc='upper left', fontsize=8)

plt.show()
# Build Model

model = ARIMA(train, order=(3, 1, 1))  

fitted = model.fit(disp=-1)  

print(fitted.summary())



# Forecast

fc, se, conf = fitted.forecast(503, alpha=0.05)  # 95% conf



# Make as pandas series

fc_series = pd.Series(fc, index=test.index)

lower_series = pd.Series(conf[:, 0], index=test.index)

upper_series = pd.Series(conf[:, 1], index=test.index)



# Plot

plt.figure(figsize=(12,5), dpi=100)

plt.plot(train, label='training')

plt.plot(test, label='actual')

plt.plot(fc_series, label='forecast')

plt.fill_between(lower_series.index, lower_series, upper_series, 

                 color='k', alpha=.15)

plt.title('Forecast vs Actuals')

plt.legend(loc='upper left', fontsize=8)

plt.show()
# Build Model

model = ARIMA(train, order=(2, 1, 1))  

fitted = model.fit(disp=-1)  

print(fitted.summary())



# Forecast

fc, se, conf = fitted.forecast(503, alpha=0.05)  # 95% conf



# Make as pandas series

fc_series = pd.Series(fc, index=test.index)

lower_series = pd.Series(conf[:, 0], index=test.index)

upper_series = pd.Series(conf[:, 1], index=test.index)



# Plot

plt.figure(figsize=(12,5), dpi=100)

plt.plot(train, label='training')

plt.plot(test, label='actual')

plt.plot(fc_series, label='forecast')

plt.fill_between(lower_series.index, lower_series, upper_series, 

                 color='k', alpha=.15)

plt.title('Forecast vs Actuals')

plt.legend(loc='upper left', fontsize=8)

plt.show()
# Build Model

model = ARIMA(train, order=(4, 1, 1))  

fitted = model.fit(disp=-1)  

print(fitted.summary())



# Forecast

fc, se, conf = fitted.forecast(503, alpha=0.05)  # 95% conf



# Make as pandas series

fc_series = pd.Series(fc, index=test.index)

lower_series = pd.Series(conf[:, 0], index=test.index)

upper_series = pd.Series(conf[:, 1], index=test.index)



# Plot

plt.figure(figsize=(12,5), dpi=100)

plt.plot(train, label='training')

plt.plot(test, label='actual')

plt.plot(fc_series, label='forecast')

plt.fill_between(lower_series.index, lower_series, upper_series, 

                 color='k', alpha=.15)

plt.title('Forecast vs Actuals')

plt.legend(loc='upper left', fontsize=8)

plt.show()
# Build Model

model = ARIMA(train, order=(4, 2, 1))  

fitted = model.fit(disp=-1)  

print(fitted.summary())



# Forecast

fc, se, conf = fitted.forecast(503, alpha=0.05)  # 95% conf



# Make as pandas series

fc_series = pd.Series(fc, index=test.index)

lower_series = pd.Series(conf[:, 0], index=test.index)

upper_series = pd.Series(conf[:, 1], index=test.index)



# Plot

plt.figure(figsize=(12,5), dpi=100)

plt.plot(train, label='training')

plt.plot(test, label='actual')

plt.plot(fc_series, label='forecast')

plt.fill_between(lower_series.index, lower_series, upper_series, 

                 color='k', alpha=.15)

plt.title('Forecast vs Actuals')

plt.legend(loc='upper left', fontsize=8)

plt.show()
# Build Model

model = ARIMA(train, order=(5, 1, 1))  

fitted = model.fit(disp=-1)  

print(fitted.summary())



# Forecast

fc, se, conf = fitted.forecast(503, alpha=0.05)  # 95% conf



# Make as pandas series

fc_series = pd.Series(fc, index=test.index)

lower_series = pd.Series(conf[:, 0], index=test.index)

upper_series = pd.Series(conf[:, 1], index=test.index)



# Plot

plt.figure(figsize=(12,5), dpi=100)

plt.plot(train, label='training')

plt.plot(test, label='actual')

plt.plot(fc_series, label='forecast')

plt.fill_between(lower_series.index, lower_series, upper_series, 

                 color='k', alpha=.15)

plt.title('Forecast vs Actuals')

plt.legend(loc='upper left', fontsize=8)

plt.show()
!pip install pmdarima
from statsmodels.tsa.arima_model import ARIMA

import pmdarima as pm

model = pm.auto_arima(df2, start_p=1, start_q=1,

                      test='adf',       # use adftest to find optimal 'd'

                      max_p=3, max_q=3, # maximum p and q

                      m=1,              # frequency of series

                      d=None,           # let model determine 'd'

                      seasonal=False,   # No Seasonality

                      start_P=0, 

                      D=0, 

                      trace=True,

                      error_action='ignore',  

                      suppress_warnings=True, 

                      stepwise=True)



print(model.summary())