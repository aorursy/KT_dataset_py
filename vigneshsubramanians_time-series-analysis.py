#Importing the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
%matplotlib inline
from statsmodels.tsa.stattools import ARMA, adfuller
from datetime import datetime

from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt


#loading the data
series = pd.Series.from_csv("../input/electricityproduction/Electric-Production.csv", header = 0)

plt.figure(figsize = (15, 7))
plt.plot(series)
plt.xlabel('Year')
plt.ylabel('Power Production in GigaWatt (GW)')
plt.show()
series.index
def stationarity(ts, window, n):
    roll_mean = ts.rolling(window).mean()
    roll_std = ts.rolling(window).std()
    
    plt.figure(figsize = (16, 8))
    
    plt.plot(ts[:n], label = 'Original Data', color = 'red')
    plt.plot(roll_mean[:n], label = 'Rolling Mean', color = 'blue')
    plt.plot(roll_std[:n], label = 'Rolling Standard Deviation', color = 'green')
    plt.title("Rolling Mean and Standard Deviation for the first %d observations"%(n))
    plt.legend(loc = 'best')
    plt.show(block = False)
    
    dftest = adfuller(ts)
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    
stationarity(series,12,len(series))
ts_log = np.log(series)
ts_log_diff = ts_log - ts_log.shift() #differencing step
ts_log_diff.dropna(inplace = True)

stationarity(ts_log_diff,365,len(ts_log_diff))
def predict(coef, history):
    yhat = 0.0
    for i in range(1, len(coef)+1):
        yhat += coef[i-1] * history[-i]
    return yhat

X = ts_log_diff.values
size = len(X) - 100
train, test = X[0:size], X[size:]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(2,0,0))
    model_fit = model.fit(trend='nc', disp=False)
    ar_coef = model_fit.arparams
    #print(ar_coef)
    yhat = predict(ar_coef, history)
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    #print('>predicted=%.3f, expected=%.3f' % (yhat, obs))
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
plt.figure(figsize=(12,8))
plt.plot(test, label = 'Original Data')
plt.plot(predictions, label = 'Predicted Data')
plt.legend(loc = 'best')
plt.show()
X = ts_log_diff.values
size = len(X) - 100
train, test = X[0:size], X[size:]
history = [x for x in train]
predictions = list()
residuals = list()

for t in range(len(test)):
    model = ARIMA(history, order=(0,0,1))
    model_fit = model.fit(trend='nc', disp=False)
    ma_coef = model_fit.maparams
    resid = model_fit.resid
    yhat = predict(ma_coef, resid)
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    #print('>predicted=%.3f, expected=%.3f' % (yhat, obs))
    error = obs - yhat # expected-predicted
    residuals.append(error)
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
plt.figure(figsize=(12,8))
plt.plot(test, label = 'Original Data')
plt.plot(predictions, label = 'Predicted Data')
plt.legend(loc = 'best')
type(X)
from statsmodels.tsa.stattools  import acf, pacf

lag_acf = acf(X, nlags = 12)
lag_pacf = pacf(X, nlags = 12)
#plt.subplot(121)
plt.figure(figsize=(12,8))
plt.plot(lag_acf)
plt.axhline(linestyle = '--', color = 'gray', y = 0)
plt.axhline(linestyle = '--', color = 'gray', y = -1.96/np.sqrt(len(X)))
plt.axhline(linestyle = '--', color = 'gray', y = 1.96/np.sqrt(len(X)))
plt.title('Autocorrelation Function')
plt.show()

#plt.subplot(122)
plt.figure(figsize=(12,8))
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(X)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(X)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.show()
type(ts_log_diff)
X = ts_log_diff.values
size = len(X) - 100
train, test = X[0:size], X[size:]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARMA(history, order=(2,0,1))
	model_fit = model.fit(trend='nc', disp=False)
	ar_coef, ma_coef = model_fit.arparams, model_fit.maparams
	resid = model_fit.resid
	yhat = predict(ar_coef, history) + predict(ma_coef, resid)
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	#print('>predicted=%.3f, expected=%.3f' % (yhat, obs))
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
plt.figure(figsize=(12,8))
plt.plot(test, label = 'Original Data')
plt.plot(predictions, label = 'Predicted Data')
plt.legend(loc = 'best')
def difference(dataset):
	diff = list()
	for i in range(1, len(dataset)):
		value = dataset[i] - dataset[i - 1]
		diff.append(value)
	return np.array(diff)
#ts_log_diff = Series.from_csv('daily-minimum-temperatures.csv', header=0)
X = series.values
size = len(X) - 100
train, test = X[0:size], X[size:]
history = [x for x in train]
#print(history)
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(1,1,1))
    model_fit = model.fit(trend='nc', disp=False)
    ar_coef, ma_coef = model_fit.arparams, model_fit.maparams
    resid = model_fit.resid
    diff = difference(history)
    yhat = history[-1] + predict(ar_coef, diff) + predict(ma_coef, resid)
    predictions.append(yhat)
    #print(t, test[t])
    obs = test[t]
    history.append(obs)
    #print('>predicted=%.3f, expected=%.3f' % (yhat, obs))
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
plt.figure(figsize=(12,8))
plt.plot(test, label = 'Original Data')
plt.plot(predictions, label = 'Predicted Data')
plt.legend(loc = 'best')
#ts_log_diff = Series.from_csv('daily-minimum-temperatures.csv', header=0)
X = ts_log_diff.values
size = len(X) - 100
train, test = X[0:size], X[size:]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(2,1,0))
	model_fit = model.fit(trend='nc', disp=False)
	ar_coef, ma_coef = model_fit.arparams, model_fit.maparams
	resid = model_fit.resid
	diff = difference(history)
	yhat = history[-1] + predict(ar_coef, diff) + predict(ma_coef, resid)
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	#print('>predicted=%.3f, expected=%.3f' % (yhat, obs))
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
plt.figure(figsize=(12,8))
plt.plot(test, label = 'Original Data')
plt.plot(predictions, label = 'Predicted Data')
plt.legend(loc = 'best')
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_log)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

#plt.subplot(411)
plt.figure(figsize = (10,7))
#plt.plot(ts_log, label='Original')
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
#plt.tight_layout()
pd.plotting.autocorrelation_plot(series)
plt.show()
from statsmodels.tsa.arima_model import ARIMA
# fit model
model = ARIMA(series, order=(5,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
# plot residual errors
residuals = pd.DataFrame(model_fit.resid)
plt.plot(residuals)
plt.show()
residuals.plot(kind='kde')
plt.show()
print(residuals.describe())
from sklearn.metrics import mean_squared_error

X = series.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(5,1,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	#print('predicted=%f, expected=%f' % (yhat, obs))
error = np.sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % error)
# plot
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()
#divide into train and validation set
train = series[:int(0.66*(len(series)))]
valid = series[int(0.66*(len(series))):]

#plotting the data
train.plot()
valid.plot()
#building the model
from pyramid.arima import auto_arima
model = auto_arima(train, trace=True, error_action='ignore', suppress_warnings=True)
model.fit(train)

forecast = model.predict(n_periods=len(valid))
forecast = pd.DataFrame(forecast,index = valid.index,columns=['Prediction'])

#plot the predictions for validation set
plt.plot(train, label='Train')
plt.plot(valid, label='Valid')
plt.plot(forecast, label='Prediction')
plt.show()
#calculate rmse
from math import sqrt
from sklearn.metrics import mean_squared_error

rms = sqrt(mean_squared_error(valid,forecast))
print(rms)