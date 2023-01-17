import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pylab as plt


# load user count timeseries
data = pd.read_csv('../input/app.csv', sep=';')
print(data.head())
# convert time column to datetime data type
data['time'] = pd.to_datetime(data['time'])
print(data.dtypes)
# use time column as index within the dataset
data = data.set_index('time')
print(data.index)
print(data.columns.tolist())
plt.figure(figsize=(20, 3))
plt.plot(data['users'], color='blue',label='User count')
plt.legend(loc='best')
moving_mean = data['users'].rolling(5).mean()
moving_std = data['users'].rolling(5).std()
plt.figure(figsize=(20, 3))
plt.plot(data['users'], color='blue',label='User count')
plt.plot(moving_mean, color='red', label='Moving mean')
plt.plot(moving_std, color='black', label = 'Moving std')
plt.legend(loc='best')
from statsmodels.tsa.stattools import adfuller

dftest = adfuller(data['users'].values)
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print(dfoutput)



ts_diff = data['users'] - data['users'].shift()
plt.figure(figsize=(20, 3))
plt.plot(data['users'], color='blue',label='User count')
plt.plot(ts_diff, color='red', label='Ts diff')
plt.legend(loc='best')
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(data['users'])

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.figure(figsize=(20, 3))
plt.plot(data['users'], color='blue',label='User count')
plt.plot(trend, color='red', label='Trend')
plt.plot(seasonal, color='black', label='Seasonality')
plt.plot(residual, color='green', label='Residuals')
plt.legend(loc='best')
from statsmodels.tsa.ar_model import AR

# fit model
model = AR(data['users'])
model_fit = model.fit()
# make prediction
pred_users = model_fit.predict(100, 300)
# plot
plt.figure(figsize=(20, 3))
plt.plot(data['users'], color='blue',label='User count')
plt.plot(pred_users, color='red', label='Prediction')
plt.legend(loc='best')

from statsmodels.tsa.arima_model import ARMA
# fit model
model = ARMA(data['users'], order=(0, 1))
model_fit = model.fit(disp=False)
# make prediction
ma_pred_users = model_fit.predict(100, 300)

plt.figure(figsize=(20, 3))
plt.plot(data['users'], color='blue',label='User count')
plt.plot(ma_pred_users, color='red',label='User count prediction')
plt.legend(loc='best')
from statsmodels.tsa.statespace.sarimax import SARIMAX
model = SARIMAX(data['users'])
model_fit = model.fit()
# make prediction
sarima_pred = model_fit.predict(100, 300)
plt.figure(figsize=(20, 3))
plt.plot(data['users'], color='blue',label='User count')
plt.plot(sarima_pred, color='red',label='User count prediction')
plt.legend(loc='best')
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
# fit model
model = SimpleExpSmoothing(data['users'])
model_fit = model.fit()
# make prediction
pred_holtwint = model_fit.predict(100, 300)
plt.figure(figsize=(20, 3))
plt.plot(data['users'], color='blue',label='User count')
plt.plot(pred_holtwint, color='red',label='User count prediction')
plt.legend(loc='best')
from statsmodels.tsa.holtwinters import ExponentialSmoothing
# fit model
model = ExponentialSmoothing(data['users'])
model_fit = model.fit()
# make prediction
hwt_pred = model_fit.predict(100, 300)
plt.figure(figsize=(20, 3))
plt.plot(data['users'], color='blue',label='User count')
plt.plot(hwt_pred, color='red',label='User count prediction')
plt.legend(loc='best')