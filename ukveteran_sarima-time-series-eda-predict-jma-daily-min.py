import numpy as np

import pandas as pd

import statsmodels.api as sm

import seaborn as sns

from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.stattools import adfuller

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

from pandas.plotting import autocorrelation_plot

from statsmodels.tsa.arima_model import ARIMA

from pandas.tseries.offsets import DateOffset



import matplotlib.pyplot as plt

%matplotlib inline



# Suppress warnings 

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/daily-min-temperatures/daily-min-temperatures.csv')

df.head()
df.columns = ['Date', 'Temp']

df.head()
df.info()

df.isnull().sum()
df['Date'] = pd.to_datetime(df['Date'])
df.set_index ('Date', inplace = True)

df.index
f, ax = plt.subplots(figsize = (16,10))

ax.plot(df, c = 'r');
df.boxplot('Temp', rot = 80, fontsize = '12',grid = True);
time_series = df['Temp']

type(time_series)
time_series.rolling(12).mean().plot(label = '12 Months Rolling Mean', figsize = (16,10))

time_series.rolling(12).std().plot(label = '12 Months Rolling Std')

time_series.plot()

plt.legend();