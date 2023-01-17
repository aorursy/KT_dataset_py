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
import pandas as pd 

import numpy as np 

data = pd.read_csv('/kaggle/input/air-passengers/AirPassengers.csv')

data.head()
data.tail()
data.dtypes
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')

data = pd.read_csv('/kaggle/input/air-passengers/AirPassengers.csv', parse_dates=['Month'], index_col='Month', date_parser=dateparse)

print('Parsed data:')

data.head()
data.index
ts = data['#Passengers']

ts.head(10)
## Some indexting techniques for timeseries data 



# Selecting a particular value in the Series object



print(ts['1949-01-01'])
# or 



from datetime import datetime 

print(ts[datetime(1949,1,1)])
# Selecting a range of values



ts[:'1949-05-01']
ts['1949']
import matplotlib.pyplot as plt



plt.plot(ts)
from statsmodels.tsa.stattools import adfuller 



def test_stationarity(timeseries):

    

    # determining rolling statistics 

    rolmean = pd.Series(timeseries).rolling(window=12).mean()

    rolstd = pd.Series(timeseries).rolling(window=12).std()

    

    # Plotting rolling statistics

    orig = plt.plot(timeseries, color='blue', label='original')

    mean = plt.plot(rolmean, color='red', label='Rolling mean')

    std = plt.plot(rolstd, color='black', label='Rolling standard deviation')

    plt.legend(loc='best')

    plt.title('Rolling mean and Standard deviation')

    plt.show(block=False)

    

    # Perform Dickey-Fuller Test

    print('Results of Dickey-Fuller Test:')

    dftest = adfuller(timeseries, autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#lags used', '#Ocservations used'])

    for key, value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

        

        print(dfoutput)
test_stationarity(ts)
plt.plot(ts)
ts_log = np.log(ts)

plt.plot(ts_log)
moving_avg = pd.Series(ts_log).rolling(window=12).mean()

plt.plot(ts_log)

plt.plot(moving_avg, color='red')
ts_log_moving_avg_diff = ts_log - moving_avg

ts_log_moving_avg_diff.head(12)
ts_log_moving_avg_diff.dropna(inplace=True)

test_stationarity(ts_log_moving_avg_diff)
expweighted_avg = ts_log.ewm(span=12).mean()

plt.plot(ts_log)

plt.plot(expweighted_avg, color='red')
ts_log_ewma_diff = ts_log - expweighted_avg

test_stationarity(ts_log_ewma_diff)
ts_log_diff = ts_log - ts_log.shift()

plt.plot(ts_log_diff)
ts_log_diff.dropna(inplace=True)

test_stationarity(ts_log_diff)
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(ts_log)



trend = decomposition.trend

seasonal = decomposition.seasonal

residual = decomposition.resid



plt.subplot(411)

plt.plot(ts_log, label='Original')

plt.legend(loc='best')

plt.subplot(412)

plt.plot(trend, label='Trend')

plt.legend(loc='best')

plt.subplot(413)

plt.plot(seasonal,label='Seasonality')

plt.legend(loc='best')

plt.subplot(414)

plt.plot(residual, label='Residuals')

plt.legend(loc='best')

plt.tight_layout()
ts_log_decompose = residual 

ts_log_decompose.dropna(inplace=True)

test_stationarity(ts_log_decompose)
## ACF and PACF plots 



from statsmodels.tsa.stattools import acf, pacf



lag_acf = acf(ts_log_diff, nlags=20)

lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')
# plot ACF 

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
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(ts_log, order=(2,1,0))

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

predictions_ARIMA_diff.head()
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()

predictions_ARIMA_diff_cumsum.head()
predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)

predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)

predictions_ARIMA_log.head()
predictions_ARIMA = np.exp(predictions_ARIMA_log)

plt.plot(ts)

plt.plot(predictions_ARIMA)

plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts)**2)/len(ts)))