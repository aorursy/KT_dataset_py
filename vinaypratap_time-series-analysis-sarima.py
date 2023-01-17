import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings("ignore")

from math import floor

data = pd.read_csv("../input/Train.csv")

train_o = data.copy()

print(data.shape)

data.columns = ['ID','Month','Passengers']

train_o.columns = ['ID','Month','Passengers']

data['Month'] = pd.to_datetime(data['Month'], format='%d-%m-%Y %H:%M')

train_o['Month'] = pd.to_datetime(train_o['Month'], format='%d-%m-%Y %H:%M')

print(data.head())

print(data.tail())
test = pd.read_csv("../input/Test.csv")

test_o = test.copy()

test.columns = ['ID','Month']

test_o.columns = ['ID','Month']

test['Month'] = pd.to_datetime(test['Month'], format='%d-%m-%Y %H:%M')

test_o['Month'] = pd.to_datetime(test_o['Month'], format='%d-%m-%Y %H:%M')

print(test.tail())

print(test.shape)  
for i in (data,train_o,test,test_o):

    i['year'] = i.Month.dt.year

    i['mon']  = i.Month.dt.month

    i['day'] = i.Month.dt.day

    i['hour'] = i.Month.dt.hour
data.head()
#Aggregating the dataset at daily level

data.index = data.Month 

data = data.resample('D').mean()
print(data.head())

print(data.tail())
#lets divide train dataset into validation and train set

#split the train data into training set and valid set

train = data['2012-08-25':'2014-06-24']

valid = data['2014-06-25':'2014-09-25']

train.shape,valid.shape
test.index = test.Month 

test = test.resample('D').mean() 
test.head()
train['Passengers'].plot(figsize=(12, 4))

plt.legend(loc='best')

plt.title('passenger traffic')

plt.show(block=False)
# data = data.assign(Passengers_Mean_Imputation=data.Passengers.fillna(data.Passengers.mean()))

# data[['Passengers_Mean_Imputation']].plot(figsize=(12, 4))

# plt.legend(loc='best')

# plt.title('Airline passenger traffic: Mean imputation')

# plt.show(block=False)
# data = data.assign(Passengers_Linear_Interpolation=data.Passengers.interpolate(method='linear'))

# data[['Passengers_Linear_Interpolation']].plot(figsize=(12, 4))

# plt.legend(loc='best')

# plt.title('Airline passenger traffic: Linear interpolation')

# plt.show(block=False)
# data['Passengers'] = data['Passengers_Linear_Interpolation']

# data.drop(columns=['Passengers_Mean_Imputation','Passengers_Linear_Interpolation'],inplace=True)
import seaborn as sns

fig = plt.subplots(figsize=(12, 2))

ax = sns.boxplot(x=train['Passengers'],whis=1.5)
fig = train.Passengers.hist(figsize = (12,4))
from pylab import rcParams

import statsmodels.api as sm

rcParams['figure.figsize'] = 12, 8

decomposition = sm.tsa.seasonal_decompose(train['Passengers'], model='additive') # additive seasonal index

fig = decomposition.plot()

plt.show()
decomposition = sm.tsa.seasonal_decompose(train['Passengers'], model='multiplicative') # multiplicative seasonal index

fig = decomposition.plot()

plt.show()
train_len = train.shape[0]

train_len
y_hat_naive = valid.copy()

y_hat_naive['naive_forecast'] = train['Passengers'][train_len-1]
plt.figure(figsize=(12,4))

plt.plot(train['Passengers'], label='Train')

plt.plot(valid['Passengers'], label='valid')

plt.plot(y_hat_naive['naive_forecast'], label='Naive forecast')

plt.legend(loc='best')

plt.title('Naive Method')

plt.show()
from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(valid['Passengers'], y_hat_naive['naive_forecast'])).round(2)

mape = np.round(np.mean(np.abs(valid['Passengers']-y_hat_naive['naive_forecast'])/valid['Passengers'])*100,2)



results = pd.DataFrame({'Method':['Naive method'], 'MAPE': [mape], 'RMSE': [rmse]})

results = results[['Method', 'RMSE', 'MAPE']]

results
y_hat_avg = valid.copy()

y_hat_avg['avg_forecast'] = train['Passengers'].mean()
plt.figure(figsize=(12,4))

plt.plot(train['Passengers'], label='Train')

plt.plot(valid['Passengers'], label='valid')

plt.plot(y_hat_avg['avg_forecast'], label='Simple average forecast')

plt.legend(loc='best')

plt.title('Simple Average Method')

plt.show()
rmse = np.sqrt(mean_squared_error(valid['Passengers'], y_hat_avg['avg_forecast'])).round(2)

mape = np.round(np.mean(np.abs(valid['Passengers']-y_hat_avg['avg_forecast'])/valid['Passengers'])*100,2)



tempResults = pd.DataFrame({'Method':['Simple average method'], 'RMSE': [rmse],'MAPE': [mape] })

results = pd.concat([results, tempResults])

results = results[['Method', 'RMSE', 'MAPE']]

results
y_hat_sma = data.copy()

ma_window = 12

y_hat_sma['sma_forecast'] = data['Passengers'].rolling(ma_window).mean()

y_hat_sma['sma_forecast'][train_len:] = y_hat_sma['sma_forecast'][train_len-1]
plt.figure(figsize=(12,4))

plt.plot(train['Passengers'], label='Train')

plt.plot(valid['Passengers'], label='valid')

plt.plot(y_hat_sma['sma_forecast'], label='Simple moving average forecast')

plt.legend(loc='best')

plt.title('Simple Moving Average Method')

plt.show()
rmse = np.sqrt(mean_squared_error(valid['Passengers'], y_hat_sma['sma_forecast'][train_len:])).round(2)

mape = np.round(np.mean(np.abs(valid['Passengers']-y_hat_sma['sma_forecast'][train_len:])/valid['Passengers'])*100,2)



tempResults = pd.DataFrame({'Method':['Simple moving average forecast'], 'RMSE': [rmse],'MAPE': [mape] })

results = pd.concat([results, tempResults])

results = results[['Method', 'RMSE', 'MAPE']]

results
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

model = SimpleExpSmoothing(train['Passengers'])

model_fit = model.fit(smoothing_level=0.2,optimized=False)

model_fit.params

y_hat_ses = valid.copy()

y_hat_ses['ses_forecast'] = model_fit.forecast(len(valid))

y_hat_ses[y_hat_ses['ses_forecast'].isnull()]

y_hat_ses['ses_forecast'].fillna(y_hat_ses['ses_forecast'].mean(),inplace=True)
plt.figure(figsize=(12,4))

plt.plot(train['Passengers'], label='Train')

plt.plot(valid['Passengers'], label='valid')

plt.plot(y_hat_ses['ses_forecast'], label='Simple exponential smoothing forecast')

plt.legend(loc='best')

plt.title('Simple Exponential Smoothing Method')

plt.show()
rmse = np.sqrt(mean_squared_error(valid['Passengers'], y_hat_ses['ses_forecast'])).round(2)

mape = np.round(np.mean(np.abs(valid['Passengers']-y_hat_ses['ses_forecast'])/valid['Passengers'])*100,2)



tempResults = pd.DataFrame({'Method':['Simple exponential smoothing forecast'], 'RMSE': [rmse],'MAPE': [mape] })

results = pd.concat([results, tempResults])

results
from statsmodels.tsa.holtwinters import ExponentialSmoothing

model = ExponentialSmoothing(np.asarray(train['Passengers']) ,seasonal_periods=12 ,trend='additive', seasonal=None)

model_fit = model.fit(smoothing_level=0.1, smoothing_slope=0.001, optimized=False)

print(model_fit.params)

y_hat_holt = valid.copy()

y_hat_holt['holt_forecast'] = model_fit.forecast(len(valid))
plt.figure(figsize=(12,4))

plt.plot( train['Passengers'], label='Train')

plt.plot(valid['Passengers'], label='valid')

plt.plot(y_hat_holt['holt_forecast'], label='Holt\'s exponential smoothing forecast')

plt.legend(loc='best')

plt.title('Holt\'s Exponential Smoothing Method')

plt.show()
rmse = np.sqrt(mean_squared_error(valid['Passengers'], y_hat_holt['holt_forecast'])).round(2)

mape = np.round(np.mean(np.abs(valid['Passengers']-y_hat_holt['holt_forecast'])/valid['Passengers'])*100,2)



tempResults = pd.DataFrame({'Method':['Holt\'s exponential smoothing method'], 'RMSE': [rmse],'MAPE': [mape] })

results = pd.concat([results, tempResults])

results = results[['Method', 'RMSE', 'MAPE']]

results
y_hat_hwa = valid.copy()

model = ExponentialSmoothing(np.asarray(train['Passengers']) ,seasonal_periods=12 ,trend='add', seasonal='add')

model_fit = model.fit(optimized=True)

print(model_fit.params)

y_hat_hwa['hw_forecast'] = model_fit.forecast(len(valid))
plt.figure(figsize=(12,4))

plt.plot( train['Passengers'], label='Train')

plt.plot(valid['Passengers'], label='valid')

plt.plot(y_hat_hwa['hw_forecast'], label='Holt Winters\'s additive forecast')

plt.legend(loc='best')

plt.title('Holt Winters\' Additive Method')

plt.show()
rmse = np.sqrt(mean_squared_error(valid['Passengers'], y_hat_hwa['hw_forecast'])).round(2)

mape = np.round(np.mean(np.abs(valid['Passengers']-y_hat_hwa['hw_forecast'])/valid['Passengers'])*100,2)



tempResults = pd.DataFrame({'Method':['Holt Winters\' additive method'], 'RMSE': [rmse],'MAPE': [mape] })

results = pd.concat([results, tempResults])

results = results[['Method', 'RMSE', 'MAPE']]

results
y_hat_hwm = valid.copy()

model = ExponentialSmoothing(np.asarray(train['Passengers']) ,seasonal_periods=2 ,trend='add', seasonal='mul')

model_fit = model.fit(optimized=True)

print(model_fit.params)

y_hat_hwm['hw_forecast'] = model_fit.forecast(len(valid))

y_hat_hwm['hw_forecast'].isnull().sum()
plt.figure(figsize=(12,4))

plt.plot( train['Passengers'], label='Train')

plt.plot(valid['Passengers'], label='valid')

plt.plot(y_hat_hwm['hw_forecast'], label='Holt Winters\'s mulitplicative forecast')

plt.legend(loc='best')

plt.title('Holt Winters\' Mulitplicative Method')

plt.show()
rmse = np.sqrt(mean_squared_error(valid['Passengers'], y_hat_hwm['hw_forecast'])).round(2)

mape = np.round(np.mean(np.abs(valid['Passengers']-y_hat_hwm['hw_forecast'])/valid['Passengers'])*100,2)



tempResults = pd.DataFrame({'Method':['Holt Winters\' multiplicative method'], 'RMSE': [rmse],'MAPE': [mape] })

results = pd.concat([results, tempResults])

results = results[['Method', 'RMSE', 'MAPE']]

results
data['Passengers'].plot(figsize=(12, 4))

plt.legend(loc='best')

plt.title('assenger traffic')

plt.show(block=False)
from statsmodels.tsa.stattools import adfuller

adf_test = adfuller(data['Passengers'])



print('ADF Statistic: %f' % adf_test[0])

print('Critical Values @ 0.05: %.2f' % adf_test[4]['5%'])

print('p-value: %f' % adf_test[1])
from statsmodels.tsa.stattools import kpss

kpss_test = kpss(data['Passengers'])



print('KPSS Statistic: %f' % kpss_test[0])

print('Critical Values @ 0.05: %.2f' % kpss_test[3]['5%'])

print('p-value: %f' % kpss_test[1])
from scipy.stats import boxcox

data_boxcox = pd.Series(boxcox(data['Passengers'], lmbda=0), index = data.index)



plt.figure(figsize=(12,4))

plt.plot(data_boxcox, label='After Box Cox tranformation')

plt.legend(loc='best')

plt.title('After Box Cox transform')

plt.show()
data_boxcox_diff = pd.Series(data_boxcox - data_boxcox.shift(1), data.index)

plt.figure(figsize=(12,4))

plt.plot(data_boxcox_diff, label='After Box Cox tranformation and differencing')

plt.legend(loc='best')

plt.title('After Box Cox transform and differencing')

plt.show()
data_boxcox_diff.dropna(inplace=True)
data_boxcox_diff.tail()
adf_test = adfuller(data_boxcox_diff)



print('ADF Statistic: %f' % adf_test[0])

print('Critical Values @ 0.05: %.2f' % adf_test[4]['5%'])

print('p-value: %f' % adf_test[1])
kpss_test = kpss(data_boxcox_diff)



print('KPSS Statistic: %f' % kpss_test[0])

print('Critical Values @ 0.05: %.2f' % kpss_test[3]['5%'])

print('p-value: %f' % kpss_test[1])
from statsmodels.graphics.tsaplots import plot_acf

plt.figure(figsize=(12,4))

plot_acf(data_boxcox_diff, ax=plt.gca(), lags = 25)

plt.show()
from statsmodels.graphics.tsaplots import plot_pacf

plt.figure(figsize=(12,4))

plot_pacf(data_boxcox_diff, ax=plt.gca(), lags = 25)

plt.show()
train_data_boxcox = data_boxcox[:train_len]

valid_data_boxcox = data_boxcox[train_len:]

train_data_boxcox_diff = data_boxcox_diff[:train_len-1]

valid_data_boxcox_diff = data_boxcox_diff[train_len-1:]
train_data_boxcox_diff
from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(train_data_boxcox_diff, order=(2, 0, 0)) 

model_fit = model.fit()

print(model_fit.params)
y_hat_ar = data_boxcox_diff.copy()

y_hat_ar['ar_forecast_boxcox_diff'] = model_fit.predict(data_boxcox_diff.index.min(), data_boxcox_diff.index.max())

y_hat_ar['ar_forecast_boxcox'] = y_hat_ar['ar_forecast_boxcox_diff'].cumsum()

y_hat_ar['ar_forecast_boxcox'] = y_hat_ar['ar_forecast_boxcox'].add(data_boxcox[0])

y_hat_ar['ar_forecast'] = np.exp(y_hat_ar['ar_forecast_boxcox'])
plt.figure(figsize=(12,4))

plt.plot(train['Passengers'], label='Train')

plt.plot(valid['Passengers'], label='valid')

plt.plot(y_hat_ar['ar_forecast'][valid.index.min():], label='Auto regression forecast')

plt.legend(loc='best')

plt.title('Auto Regression Method')

plt.show()
rmse = np.sqrt(mean_squared_error(valid['Passengers'], y_hat_ar['ar_forecast'][valid.index.min():])).round(2)

mape = np.round(np.mean(np.abs(valid['Passengers']-y_hat_ar['ar_forecast'][valid.index.min():])/valid['Passengers'])*100,2)



tempResults = pd.DataFrame({'Method':['Autoregressive (AR) method'], 'RMSE': [rmse],'MAPE': [mape] })

results = pd.concat([results, tempResults])

results = results[['Method', 'RMSE', 'MAPE']]

results
model = ARIMA(train_data_boxcox_diff, order=(0, 0, 2)) 

model_fit = model.fit()

print(model_fit.params)
y_hat_ma = data_boxcox_diff.copy()

y_hat_ma['ma_forecast_boxcox_diff'] = model_fit.predict(data_boxcox_diff.index.min(), data_boxcox_diff.index.max())

y_hat_ma['ma_forecast_boxcox'] = y_hat_ma['ma_forecast_boxcox_diff'].cumsum()

y_hat_ma['ma_forecast_boxcox'] = y_hat_ma['ma_forecast_boxcox'].add(data_boxcox[0])

y_hat_ma['ma_forecast'] = np.exp(y_hat_ma['ma_forecast_boxcox'])
plt.figure(figsize=(12,4))

plt.plot(data['Passengers'][:train_len], label='Train')

plt.plot(data['Passengers'][train_len:], label='valid')

plt.plot(y_hat_ma['ma_forecast'][valid.index.min():], label='Moving average forecast')

plt.legend(loc='best')

plt.title('Moving Average Method')

plt.show()
rmse = np.sqrt(mean_squared_error(valid['Passengers'], y_hat_ma['ma_forecast'][valid.index.min():])).round(2)

mape = np.round(np.mean(np.abs(valid['Passengers']-y_hat_ma['ma_forecast'][valid.index.min():])/valid['Passengers'])*100,2)



tempResults = pd.DataFrame({'Method':['Moving Average (MA) method'], 'RMSE': [rmse],'MAPE': [mape] })

results = pd.concat([results, tempResults])

results = results[['Method', 'RMSE', 'MAPE']]

results
model = ARIMA(train_data_boxcox_diff, order=(1, 0, 1))

model_fit = model.fit()

print(model_fit.params)
y_hat_arma = data_boxcox_diff.copy()

y_hat_arma['arma_forecast_boxcox_diff'] = model_fit.predict(data_boxcox_diff.index.min(), data_boxcox_diff.index.max())

y_hat_arma['arma_forecast_boxcox'] = y_hat_arma['arma_forecast_boxcox_diff'].cumsum()

y_hat_arma['arma_forecast_boxcox'] = y_hat_arma['arma_forecast_boxcox'].add(data_boxcox[0])

y_hat_arma['arma_forecast'] = np.exp(y_hat_arma['arma_forecast_boxcox'])
plt.figure(figsize=(12,4))

plt.plot( data['Passengers'][:train_len-1], label='Train')

plt.plot(data['Passengers'][train_len-1:], label='valid')

plt.plot(y_hat_arma['arma_forecast'][valid.index.min():], label='ARMA forecast')

plt.legend(loc='best')

plt.title('ARMA Method')

plt.show()
rmse = np.sqrt(mean_squared_error(valid['Passengers'], y_hat_arma['arma_forecast'][train_len-1:])).round(2)

mape = np.round(np.mean(np.abs(valid['Passengers']-y_hat_arma['arma_forecast'][train_len-1:])/valid['Passengers'])*100,2)



tempResults = pd.DataFrame({'Method':['Autoregressive moving average (ARMA) method'], 'RMSE': [rmse],'MAPE': [mape] })

results = pd.concat([results, tempResults])

results = results[['Method', 'RMSE', 'MAPE']]

results
model = ARIMA(train_data_boxcox, order=(2, 1, 4))

model_fit = model.fit()

print(model_fit.params)
y_hat_arima = data_boxcox_diff.copy()

y_hat_arima['arima_forecast_boxcox_diff'] = model_fit.predict(data_boxcox_diff.index.min(), data_boxcox_diff.index.max())

y_hat_arima['arima_forecast_boxcox'] = y_hat_arima['arima_forecast_boxcox_diff'].cumsum()

y_hat_arima['arima_forecast_boxcox'] = y_hat_arima['arima_forecast_boxcox'].add(data_boxcox[0])

y_hat_arima['arima_forecast'] = np.exp(y_hat_arima['arima_forecast_boxcox'])
plt.figure(figsize=(12,4))

plt.plot(train['Passengers'], label='Train')

plt.plot(valid['Passengers'], label='valid')

plt.plot(y_hat_arima['arima_forecast'][valid.index.min():], label='ARIMA forecast')

plt.legend(loc='best')

plt.title('Autoregressive integrated moving average (ARIMA) method')

plt.show()
rmse = np.sqrt(mean_squared_error(valid['Passengers'], y_hat_arima['arima_forecast'][valid.index.min():])).round(2)

mape = np.round(np.mean(np.abs(valid['Passengers']-y_hat_arima['arima_forecast'][valid.index.min():])/valid['Passengers'])*100,2)



tempResults = pd.DataFrame({'Method':['Autoregressive integrated moving average (ARIMA) method'], 'RMSE': [rmse],'MAPE': [mape] })

results = pd.concat([results, tempResults])

results = results[['Method', 'RMSE', 'MAPE']]

results
from statsmodels.tsa.statespace.sarimax import SARIMAX



model = SARIMAX(train_data_boxcox, order=(3, 1, 4), seasonal_order=(0, 1, 2, 7)) 

model_fit = model.fit()

print(model_fit.params)
y_hat_sarima = data_boxcox_diff.copy()

y_hat_sarima['sarima_forecast_boxcox'] = model_fit.predict(data_boxcox_diff.index.min(), data_boxcox_diff.index.max())

y_hat_sarima['sarima_forecast'] = np.exp(y_hat_sarima['sarima_forecast_boxcox'])
plt.figure(figsize=(12,4))

plt.plot(train['Passengers'], label='Train')

plt.plot(valid['Passengers'], label='valid')

plt.plot(y_hat_sarima['sarima_forecast'][valid.index.min():], label='SARIMA forecast')

plt.legend(loc='best')

plt.title('Seasonal autoregressive integrated moving average (SARIMA) method')

plt.show()
rmse = np.sqrt(mean_squared_error(valid['Passengers'], y_hat_sarima['sarima_forecast'][valid.index.min():])).round(2)

mape = np.round(np.mean(np.abs(valid['Passengers']-y_hat_sarima['sarima_forecast'][valid.index.min():])/valid['Passengers'])*100,2)



tempResults = pd.DataFrame({'Method':['Seasonal autoregressive integrated moving average (SARIMA) method'], 'RMSE': [rmse],'MAPE': [mape] })

results = pd.concat([results, tempResults])

results = results[['Method', 'RMSE', 'MAPE']]

results
#calculate hourly rate of passengers

train_o['ratio'] = train_o['Passengers']/train_o['Passengers'].sum()

#group the hourly ratio

temp=train_o.groupby(['hour'])['ratio'].sum()

temp= pd.DataFrame(temp,columns=['ratio']).reset_index()

temp.head()
#y_hat_sarima_t = test.copy()

predict = model_fit.predict(test.index.min(), test.index.max())

predict = np.exp(predict)

test['prediction'] = predict

merge = pd.merge(test,test_o,on=('day','mon','year'),how='left')

merge['hour'] = merge['hour_y']

merge = merge.drop(['year','mon','hour_x','hour_y'],axis=1)

prediction = pd.merge(merge,temp,on='hour',how='left')

prediction.head()

prediction['Count'] = prediction['prediction']*prediction['ratio']*24

prediction['ID'] = prediction['ID_y']

prediction.head()
submission = prediction.drop(['ID_x','day','prediction','ID_y','Month','hour','ratio'],axis=1)

pd.DataFrame(submission,columns=['ID','Count']).to_csv("Submission_SARIMA.csv",index=False)