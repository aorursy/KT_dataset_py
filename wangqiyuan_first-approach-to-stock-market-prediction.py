import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # Plotting



# Read data and make Date column index

djia = pd.read_csv("../input/DJIA_table.csv", parse_dates=['Date'], index_col='Date')

print(djia.head())
ts = djia['Open']

ts = ts.head(100)

plt.plot(ts)
from statsmodels.tsa.stattools import adfuller

def test_stationarity(tieseries):



    rollmean = tieseries.rolling(window=12).mean()

    rollstd = tieseries.rolling(window=12).std()



    plt.plot(tieseries, color="blue", label="Original")

    plt.plot(rollmean, color="red", label="Rolling Mean")

    plt.plot(rollstd, color="black", label="Rolling Std")    

    plt.legend(loc='best')

    plt.title('Rolling Mean & Standard Deviation')

    plt.show(block=False)



    #Perform Dickey-Fuller test:

    print('Results of Dickey-Fuller Test:')

    dftest = adfuller(tieseries, autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    print(dfoutput,'%f' % (1/10**8))
test_stationarity(ts)
import matplotlib.gridspec as gridspec



ts_log = np.log(ts)

fig = plt.figure(constrained_layout = True)

gs_1 = gridspec.GridSpec(2, 3, figure = fig)

ax_1 = fig.add_subplot(gs_1[0, :])

ax_1.plot(ts_log)

ax_1.set_xlabel('time')

ax_1.set_ylabel('data')

plt.title('Logged time serie')



ax_2 = fig.add_subplot(gs_1[1, :])

ax_2.plot(ts)

ax_1.set_xlabel('time')

ax_1.set_ylabel('data')

plt.title('Original time serie')
from sklearn import datasets, linear_model



ts_wi = ts_log.reset_index()

df_values = ts_wi.values

train_y = df_values[:,1]

train_y = train_y[:, np.newaxis]

train_x = ts_wi.index

train_x = train_x[:, np.newaxis]

regr = linear_model.LinearRegression()

regr.fit(train_x, train_y)

pred = regr.predict(train_x)

plt.plot(ts_wi.Date, pred)

plt.plot(ts_log)
mov_average = ts_log.rolling(12).mean()

plt.plot(mov_average)

plt.plot(ts_log)
ts_log_mov_av_diff = ts_log - mov_average

#ts_log_mov_av_diff.head(12)

ts_log_mov_av_diff.dropna(inplace=True)



test_stationarity(ts_log_mov_av_diff)
ts_log_mov_reg_diff = ts_log - pred[:,0]

#ts_log_mov_av_diff.head(12)

ts_log_mov_reg_diff.dropna(inplace=True)



test_stationarity(ts_log_mov_reg_diff)
from statsmodels.graphics import tsaplots as tsa

ts_log_diff = ts_log - ts_log.shift(1)

fig = plt.figure(figsize=(12,8))

ax1 = fig.add_subplot(211)

fig = tsa.plot_acf(ts_log_diff.iloc[13:], lags=40, ax=ax1)

ax2 = fig.add_subplot(212)

fig = tsa.plot_pacf(ts_log_diff.iloc[13:], lags=40, ax=ax2)
ts_log_diff.dropna(inplace=True)

test_stationarity(ts_log_diff)
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(ts_log, freq=4, model='additive')



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
#ts_decompose = residual

ts_decompose = ts_log_diff

ts_decompose.dropna(inplace=True)

test_stationarity(ts_decompose)
fig = plt.figure(figsize=(12,8))

ax1 = fig.add_subplot(211)

fig = tsa.plot_acf(ts_decompose, lags=40, ax=ax1)

ax2 = fig.add_subplot(212)

fig = tsa.plot_pacf(ts_decompose, lags=40, ax=ax2)
from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.statespace.sarimax import SARIMAX



# Raw time serie

fig = plt.figure(constrained_layout=True) 

gs = gridspec.GridSpec(2, 1, figure=fig)

ax = fig.add_subplot(gs[0, :])

ax.plot(ts_decompose)

ax.set_xlabel('time [months]')

ax.set_ylabel('data')

ax.set_title('Logged data')



model = ARIMA(ts_decompose, order=(1, 0, 0))

res = model.fit(disp=-2)



ax2 = fig.add_subplot(gs[1, :])

ax2.plot(res.fittedvalues)

ax2.set_xlabel('time [months]')

ax2.set_ylabel('data')

ax2.set_title('ARIMA model')



print('ARIMA RMSE: %.6f'% np.sqrt(sum((res.fittedvalues-ts_decompose)**2)/len(ts)))
mod = SARIMAX(ts_decompose, trend='n', order=(1,1,0), seasonal_order=(3,0,3,4))

resSARIMAX = mod.fit()

pred = resSARIMAX.predict()



# Raw time serie

fig = plt.figure(constrained_layout=True) 

gs = gridspec.GridSpec(2, 1, figure=fig)

ax = fig.add_subplot(gs[0, :])

ax.plot(ts_decompose)

ax.set_xlabel('time [months]')

ax.set_ylabel('data')

ax.set_title('Logged data')



ax2 = fig.add_subplot(gs[1, :])

ax2.plot(pred)

ax2.set_xlabel('time [months]')

ax2.set_ylabel('data')

ax2.set_title('SARIMAX model')

print('SARIMAX RMSE: %.6f'% np.sqrt(sum((pred-ts_decompose)**2)/len(ts)))
predictions_SARIMA_diff = pd.Series(pred, copy=True)

predictions_SARIMA_diff_cumsum = predictions_SARIMA_diff.cumsum()

predictions_SARIMA_log = pd.Series(ts_log.iloc[0], index=ts_log.index)

predictions_SARIMA_log = predictions_SARIMA_log.add(predictions_SARIMA_diff_cumsum,fill_value=0)

predictions_SARIMA = np.exp(predictions_SARIMA_log)



predictions_ARIMA_diff = pd.Series(res.fittedvalues, copy=True)

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()

predictions_ARIMA_log = pd.Series(ts_log.iloc[0], index=ts_log.index)

predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)

predictions_ARIMA = np.exp(predictions_ARIMA_log)

plt.plot(predictions_ARIMA)

print('ARIMA RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts)**2)/len(ts)))



plt.plot(ts)

plt.plot(predictions_SARIMA)

print('SARIMA RMSE: %.4f'% np.sqrt(sum((predictions_SARIMA-ts)**2)/len(ts)))