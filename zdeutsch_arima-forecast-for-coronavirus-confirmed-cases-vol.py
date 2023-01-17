import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

df = df.fillna('unknow')

df.head()
df = df.append(df.sum(numeric_only=True), ignore_index=True)

df
df = df.iloc[df.shape[0] - 1][4:df.shape[1]]

df
split_point = len(df) - 4

dataset, validation = df[0:split_point], df[split_point:]

print('Dataset %d, Validation %d' % (len(dataset), len(validation)))

dataset.to_csv('dataset.csv', header = False)

validation.to_csv('validation.csv', header = False)
from sklearn.metrics import mean_squared_error

from math import sqrt

# load data

series = pd.read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)

# prepare data

X = series.values

X = X.astype('float32')

train_size = int(len(X) * 0.50)

train, test = X[0:train_size], X[train_size:]

# walk-forward validation

history = [x for x in train]

predictions = list()

for i in range(len(test)):

	# predict

	yhat = history[-1]

	predictions.append(yhat)

	# observation

	obs = test[i]

	history.append(obs)

	print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))

# report performance

rmse = sqrt(mean_squared_error(test, predictions))

print('RMSE: %.3f' % rmse)
series.describe()
series.plot()
plt.figure(1)

plt.subplot(211)

series.hist()

plt.subplot(212)

series.plot(kind='kde')

plt.show()
from statsmodels.tsa.stattools import adfuller



# create a differenced time series

def difference(dataset):

	diff = list()

	for i in range(1, len(dataset)):

		value = dataset[i] - dataset[i - 1]

		diff.append(value)

	return pd.Series(diff)



# difference data

stationary = difference(X)

stationary.index = series.index[1:]

# check if stationary

result = adfuller(stationary)

print('ADF Statistic: %f' % result[0])

print('p-value: %f' % result[1])

print('Critical Values:')

for key, value in result[4].items():

	print('\t%s: %.3f' % (key, value))

# save

stationary.to_csv('stationary.csv', header = False)
from statsmodels.graphics.tsaplots import plot_acf

from statsmodels.graphics.tsaplots import plot_pacf



plt.figure()

plt.subplot(211)

plot_acf(series, lags=24, ax=plt.gca())

plt.subplot(212)

plot_pacf(series, lags=24, ax=plt.gca())

plt.show()
import warnings

from statsmodels.tsa.arima_model import ARIMA

from sklearn.metrics import mean_squared_error

from math import sqrt



# evaluate an ARIMA model for a given order (p,d,q) and return RMSE

def evaluate_arima_model(X, arima_order):

	# prepare training dataset

	X = X.astype('float32')

	train_size = int(len(X) * 0.50)

	train, test = X[0:train_size], X[train_size:]

	history = [x for x in train]

	# make predictions

	predictions = list()

	for t in range(len(test)):

		model = ARIMA(history, order=arima_order)

		model_fit = model.fit(disp=0)

		yhat = model_fit.forecast()[0]

		predictions.append(yhat)

		history.append(test[t])

	# calculate out of sample error

	rmse = sqrt(mean_squared_error(test, predictions))

	return rmse



# evaluate combinations of p, d and q values for an ARIMA model

def evaluate_models(dataset, p_values, d_values, q_values):

	dataset = dataset.astype('float32')

	best_score, best_cfg = float("inf"), None

	for p in p_values:

		for d in d_values:

			for q in q_values:

				order = (p,d,q)

				try:

					rmse = evaluate_arima_model(dataset, order)

					if rmse < best_score:

						best_score, best_cfg = rmse, order

					print('ARIMA%s RMSE=%.3f' % (order,rmse))

				except:

					continue

	print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))



# evaluate parameters

p_values = range(0,13)

d_values = range(0, 4)

q_values = range(0, 13)

warnings.filterwarnings("ignore")

evaluate_models(X, p_values, d_values, q_values)
best_cfg = (0, 1, 0)
train_size = int(len(X) * 0.50)

train, test = X[0:train_size], X[train_size:]

# walk-forward validation

history = [x for x in train]

predictions = list()

for i in range(len(test)):

	# predict

	model = ARIMA(history, order=best_cfg)

	model_fit = model.fit(disp=0)

	yhat = model_fit.forecast()[0]

	predictions.append(yhat)

	# observation

	obs = test[i]

	history.append(obs)

# errors

residuals = [test[i]-predictions[i] for i in range(len(test))]

residuals = pd.DataFrame(residuals)

plt.figure()

plt.subplot(211)

residuals.hist(ax=plt.gca())

plt.subplot(212)

residuals.plot(kind='kde', ax=plt.gca())

plt.show()
from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.arima_model import ARIMAResults

from scipy.stats import boxcox

from sklearn.metrics import mean_squared_error

from math import sqrt

from math import exp

from math import log

import numpy



# load and prepare datasets

dataset = pd.read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)

X = dataset.values.astype('float32')

history = [x for x in X]

validation = pd.read_csv('validation.csv', header=None, index_col=0, parse_dates=True, squeeze=True)

y = validation.values.astype('float32')

# run model

model = ARIMA(history, order=best_cfg)

model_fit = model.fit(disp=0)

# make first prediction

predictions = list()

yhat = model_fit.forecast()[0][0]

predictions.append(yhat)

history.append(y[0])

print('>Predicted=%.3f, Expected=%3.f' % (yhat, y[0]))

# rolling forecasts

for i in range(1, len(y)):

  # predict

  model = ARIMA(history, order=best_cfg)

  model_fit = model.fit(disp=0)

  yhat = model_fit.forecast()[0][0]

  predictions.append(yhat)

  # observation

  obs = y[i]

  history.append(obs)

  print('>Predicted=%i, Expected=%i' % (yhat, obs))

# report performance

rmse = sqrt(mean_squared_error(y, predictions))

print('RMSE: %i' % rmse)

#predict next day

# predict

model = ARIMA(history, order=(0,1,0))

model_fit = model.fit(disp=0)

yhat = model_fit.forecast()[0][0]

predictions.append(yhat)

print('>Predicted next day volume=%i' % (yhat))

plt.plot(y)

plt.plot(predictions, color='red')

plt.show()