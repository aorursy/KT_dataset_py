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
df = pd.read_csv('../input/ovary-cancer-data/OvaryCancer.csv')

df.head()
df.groupby(['stage'])

df
df1=df.groupby('stage').groups

df1
df2=df.groupby('operation').groups

df2
plt.clf()

df.groupby('stage').size().plot(kind='bar')

plt.show()
plt.clf()

df.groupby('operation').size().plot(kind='bar')

plt.show()
plt.clf()

df.groupby('stage').sum().plot(kind='bar')

plt.show()
plt.clf()

df.groupby('operation').sum().plot(kind='bar')

plt.show()