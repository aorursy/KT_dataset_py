import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import statsmodels.api as sm

import seaborn as sb

sb.set_style('darkgrid')

path = '../input/1950csv/1950.csv'

stock_data = pd.read_csv(path)

# stock_data['Date'] = stock_data['Date'].to_dates='coerce'

stock_data = stock_data.sort_index(by='Date')

stock_data = stock_data.set_index('Date')
stock_data['Close'].plot(figsize=(16, 12))
stock_data['First Difference'] = stock_data['Close'] - stock_data['Close'].shift()

stock_data['First Difference'].plot(figsize=(16, 12))
stock_data['Natural Log'] = stock_data['Close'].apply(lambda x: np.log(x))

stock_data['Natural Log'].plot(figsize=(16, 12))


stock_data['Original Variance'] = pd.Series.rolling(stock_data['Close'],window=30,center=True).var()

stock_data['Log Variance'] = pd.Series.rolling(stock_data['Natural Log'],window=30,center=True).var()



fig, ax = plt.subplots(2, 1, figsize=(13, 12))

stock_data['Original Variance'].plot(ax=ax[0], title='Original Variance')

stock_data['Log Variance'].plot(ax=ax[1], title='Log Variance')

fig.tight_layout()
stock_data['Logged First Difference'] = stock_data['Natural Log'] - stock_data['Natural Log'].shift()

stock_data['Logged First Difference'].plot(figsize=(16, 12))
stock_data['Lag 1'] = stock_data['Logged First Difference'].shift()

stock_data['Lag 2'] = stock_data['Logged First Difference'].shift(2)

stock_data['Lag 5'] = stock_data['Logged First Difference'].shift(5)

stock_data['Lag 30'] = stock_data['Logged First Difference'].shift(30)

sb.jointplot('Logged First Difference', 'Lag 1', stock_data, kind='reg', height=13)
from statsmodels.tsa.stattools import acf

from statsmodels.tsa.stattools import pacf



lag_correlations = acf(stock_data['Logged First Difference'].iloc[1:])

lag_partial_correlations = pacf(stock_data['Logged First Difference'].iloc[1:])
fig, ax = plt.subplots(figsize=(16,12))

ax.plot(lag_correlations, marker='o', linestyle='--')

ax.plot(lag_partial_correlations, marker='o', linestyle='--')
from statsmodels.tsa.seasonal import seasonal_decompose



decomposition = seasonal_decompose(stock_data['Natural Log'], model='additive', freq=30)

fig = plt.figure()

fig = decomposition.plot()
co2_data = sm.datasets.co2.load_pandas().data

co2_data.co2.interpolate(inplace=True)

result = sm.tsa.seasonal_decompose(co2_data.co2)

fig = plt.figure()

fig = result.plot()
model = sm.tsa.ARIMA(stock_data['Natural Log'].iloc[1:], order=(1, 0, 0))

results = model.fit(disp=-1)

stock_data['Forecast'] = results.fittedvalues

stock_data[['Natural Log', 'Forecast']].plot(figsize=(16, 12))
model = sm.tsa.ARIMA(stock_data['Logged First Difference'].iloc[1:], order=(1, 0, 0))

results = model.fit(disp=-1)

stock_data['Forecast'] = results.fittedvalues

stock_data[['Logged First Difference', 'Forecast']].plot(figsize=(16, 12))
stock_data[['Logged First Difference', 'Forecast']].iloc[100:200, : ].plot(figsize=(16, 12))
model = sm.tsa.ARIMA(stock_data['Logged First Difference'].iloc[1:], order=(0, 0, 1))

results = model.fit(disp=-1)

stock_data['Forecast'] = results.fittedvalues

stock_data[['Logged First Difference', 'Forecast']].plot(figsize=(16, 12))