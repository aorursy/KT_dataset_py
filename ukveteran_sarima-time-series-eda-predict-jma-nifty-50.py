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
df = pd.read_csv('../input/nifty50-stock-market-data/ADANIPORTS.csv')

df.head()
df.columns = ['Date',"", "","","","","","","","",'Val',"","","",""]

df.head()
df.info()

df.isnull().sum()
df['Date'] = pd.to_datetime(df['Date'])
df.set_index ('Date', inplace = True)

df.index
df_new = df['1998-01-01':]

df_new.tail()
df_new.describe().transpose()
time_series = df_new['Val']

type(time_series)
time_series.rolling(12).mean().plot(label = '12 Months Rolling Mean', figsize = (16,10))

time_series.rolling(12).std().plot(label = '12 Months Rolling Std')

time_series.plot()

plt.legend();
df1 = pd.read_csv('../input/nifty50-stock-market-data/ASIANPAINT.csv')

df1.head()
df1.columns = ['Date',"", "","","","","","","","",'Val',"","","",""]

df1.head()
df1.info()

df1.isnull().sum()
df1['Date'] = pd.to_datetime(df1['Date'])
df1.set_index ('Date', inplace = True)

df1.index
df1_new = df1['1998-01-01':]

df1_new.tail()
df1_new.describe().transpose()
df1_new.boxplot('Val', rot = 80, fontsize = '12',grid = True);
time_series = df1_new['Val']

type(time_series)
time_series.rolling(12).mean().plot(label = '12 Months Rolling Mean', figsize = (16,10))

time_series.rolling(12).std().plot(label = '12 Months Rolling Std')

time_series.plot()

plt.legend();