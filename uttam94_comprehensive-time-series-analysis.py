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
#Importing Librabries 

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns 



# TIME SERIES

from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.statespace.sarimax import SARIMAX

from pandas.plotting import autocorrelation_plot

from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic

import statsmodels.formula.api as smf

import statsmodels.tsa.api as smt

import statsmodels.api as sm

import scipy.stats as scs



# warnings 

import warnings

warnings.filterwarnings("ignore")
# Now, we will load the data set and look at some initial rows and data types of the columns:

data = pd.read_csv('../input/AirPassengers.csv')

print (data.head())

print ('\n Data Types:')

print (data.dtypes)
# The data contains a particular month and number of passengers travelling in that month. In order to read the data as a time series, we have to pass special arguments to the read_csv command:

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')

data = pd.read_csv('../input/AirPassengers.csv', parse_dates=['Month'], index_col='Month',date_parser=dateparse)

print ('\n Parsed Data:')

print (data.head())

print (data.dtypes)
data.index
ts= data['#Passengers']
ts[:'1949-05-01']
plt.plot(ts)
def test_stationarity(timeseries):

    #Determing rolling statistics

    plt.figure(figsize=(16,6))

    plt.plot(timeseries, color='blue',label='Original')

    plt.plot(timeseries.rolling(window= 12,center= False).mean(),label='Rolling Mean')

    plt.plot(timeseries.rolling(window=12,center= False).std(),label='Rolling std')

    plt.legend()
test_stationarity(ts)
# Stationarity tests

def Dickey_Fuller(timeseries):

    

    #Perform Dickey-Fuller test:

    print('Results of Dickey-Fuller Test:')

    dftest = adfuller(timeseries, autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    print (dfoutput)



Dickey_Fuller(ts)
ts_log = np.log(ts)

plt.plot(ts_log)
moving_avg = ts_log.rolling(12).mean()

plt.plot(ts_log)

plt.plot(moving_avg, color='red')
ts_log_moving_avg_diff = ts_log - moving_avg

ts_log_moving_avg_diff.head(12)
ts_log_moving_avg_diff.dropna(inplace=True)

test_stationarity(ts_log_moving_avg_diff)
Dickey_Fuller(ts_log_moving_avg_diff)
expwighted_avg = ts_log.ewm(halflife=12).mean()

plt.plot(ts_log)

plt.plot(expwighted_avg, color='red')
ts_log_ewma_diff = ts_log - expwighted_avg
Dickey_Fuller(ts_log_ewma_diff)
ts_log_diff = ts_log - ts_log.shift()

plt.plot(ts_log_diff)
ts_log_diff.dropna(inplace=True)

test_stationarity(ts_log_diff)

print(Dickey_Fuller(ts_log_diff))
decomposition = sm.tsa.seasonal_decompose(ts_log)



trend = decomposition.trend

seasonal = decomposition.seasonal

residual = decomposition.resid
import statsmodels.api as sm

tse = sm.tsa.seasonal_decompose(ts_log.values,freq=12,model='multiplicative')

tse.plot()
tse = sm.tsa.seasonal_decompose(ts_log.values,freq=12,model='additive')

tse.plot()
ts_log_decompose = residual

ts_log_decompose.dropna(inplace=True)

test_stationarity(ts_log_decompose)

print(Dickey_Fuller(ts_log_decompose))
lag_acf = acf(ts_log_diff, nlags=20)

lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')
#Plot ACF: 

plt.subplot(121) 

plt.plot(lag_acf)

plt.axhline(y=0,linestyle='--',color='gray')

plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')

plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')

plt.title('Autocorrelation Function')



#Plot PACF:

plt.subplot(122)

plt.plot(lag_pacf)

plt.axhline(y=0,linestyle='--',color='gray')

plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')

plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')

plt.title('Partial Autocorrelation Function')

plt.tight_layout()
model = ARIMA(ts_log, order=(2, 1, 0))  

results_AR = model.fit(disp=-1)  

plt.plot(ts_log_diff)

plt.plot(results_AR.fittedvalues, color='red')

plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-ts_log_diff)**2))
model = ARIMA(ts_log, order=(0, 1, 2))  

results_MA = model.fit(disp=-1)  

plt.plot(ts_log_diff)

plt.plot(results_MA.fittedvalues, color='red')

plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-ts_log_diff)**2))
model = ARIMA(ts_log, order=(2, 1, 2))  

results_ARIMA = model.fit(disp=-1)  

plt.plot(ts_log_diff)

plt.plot(results_ARIMA.fittedvalues, color='red')

plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)

print(predictions_ARIMA_diff.head())
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()

print(predictions_ARIMA_diff_cumsum.head())
predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)

predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)

predictions_ARIMA_log.head()
predictions_ARIMA = np.exp(predictions_ARIMA_log)

plt.plot(ts)

plt.plot(predictions_ARIMA)

plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts)**2)/len(ts)))
from fbprophet import Prophet
df = pd.read_csv('../input/AirPassengers.csv')

df.head()

print(df.dtypes)
#converting month into date time object 

df['Month'] = pd.DatetimeIndex(df['Month'])

df.dtypes
# renaming columns according to prophet model

df = df.rename(columns={'Month': 'ds',

                        '#Passengers': 'y'})

df.head(5)
ax = df.set_index('ds').plot(figsize=(12, 8))

ax.set_ylabel('Monthly Number of Airline Passengers')

ax.set_xlabel('Date')

plt.show()
#set the uncertainty interval to 95% (the Prophet default is 80%)

my_model = Prophet(interval_width=0.95)
my_model.fit(df)
future_dates = my_model.make_future_dataframe(periods=36, freq='MS')

future_dates.tail()
forecast = my_model.predict(future_dates)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
my_model.plot(forecast,

              uncertainty=True)
my_model.plot_components(forecast)
from fbprophet.diagnostics import cross_validation

df_cv = cross_validation(my_model, initial='730 days', period='180 days', horizon = '90 days')

df_cv.head()
from fbprophet.diagnostics import performance_metrics

df_p = performance_metrics(df_cv)

df_p.head()