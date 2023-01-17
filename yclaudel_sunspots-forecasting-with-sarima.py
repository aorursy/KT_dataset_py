# Import librairies

%matplotlib inline 

import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import statsmodels.api as sm

plt.style.use('seaborn')

plt.rcParams['figure.figsize'] = [16, 9]

from statsmodels.tsa import stattools

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from timeit import default_timer as timers
# load the data

path = "./kaggle/input/daily-sun-spot-data-1818-to-2019/"

filename = os.path.join(path,"sunspot_data.csv")

df = pd.read_csv('/kaggle/input/daily-sun-spot-data-1818-to-2019/sunspot_data.csv', delimiter=',', na_values=['-1'])

df.dataframeName = 'sunspot_data.csv'

del(df['Unnamed: 0'])

df.columns = ['year', 'month', 'day', 'fraction','sunspots', 'sdt', 'obs','indicator']

df.head(-5)



# Add the column time 

df['time']=df[['year', 'month', 'day']].apply(lambda s: pd.datetime(*s),axis = 1)

# time column is the index of the dataframe

df.index = df['time']

# replace the Nan by linear interpolation 

df['sunspots'].interpolate(method='linear', inplace=True)
ts = pd.Series(data=df.sunspots, index=df.index)

#ts = ts['1900-01-01':]

ts_month = ts.resample('MS').mean()

ts_quarter = ts.resample('Q').mean()

ts_quarter.plot()

plt.show()
plot_pacf(ts_quarter,lags=100,title='Sunspots')

plt.show()
plot_acf(ts_quarter,lags=100,title='Sunspots')

plt.show()
from statsmodels.tsa.stattools import adfuller

def printADFTest(serie):

    result = adfuller(serie, autolag='AIC')

    print("ADF Statistic %F" % (result[0]))

    print(f'p-value: {result[1]}')

    for key, value in result[4].items():

        print('Critial Values:')

        print(f'   {key}, {value}')

    print('\n')



#d = 0

printADFTest(ts_quarter)

#d = 1 

#printADFTest(ts_quarter.diff(1).dropna())
model = sm.tsa.statespace.SARIMAX(ts_quarter, trend='n', order=(3,0,10), seasonal_order=(1,1,0,43))

results = model.fit()

print(results.summary())

forecast = results.predict(start = ts_quarter.index[-2], end= ts_quarter.index[-2] + pd.DateOffset(months=240), dynamic= True) 

ts_quarter.plot()

forecast.plot()

plt.show()
!pip install pmdarima
import pmdarima as pm

grid_model = pm.auto_arima(ts_quarter, start_p=1, start_q=1,

                         test='adf',

                         max_p=4, max_q=4, m=43,

                         start_P=0, seasonal=True,

                         d=0, D=1, trace=True,

                         error_action='ignore',  

                         suppress_warnings=True, 

                         stepwise=True)

print(grid_model.summary())
period = 60

fitted, confint = grid_model.predict(n_periods=period, return_conf_int=True)

index_of_fc = pd.date_range(ts_quarter.index[-1], periods = period, freq='Q')



# make series for plotting purpose

fitted_series = pd.Series(fitted, index=index_of_fc)

lower_series = pd.Series(confint[:, 0], index=index_of_fc)

upper_series = pd.Series(confint[:, 1], index=index_of_fc)



# Plot

plt.plot(ts_quarter)

plt.plot(fitted_series, color='darkgreen')

plt.fill_between(lower_series.index, 

                 lower_series, 

                 upper_series, 

                 color='k', alpha=.15)



plt.title("SARIMA - Forecast Sunspots")

plt.show()