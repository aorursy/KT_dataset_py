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
df = pd.read_csv('../input/freedom-of-speech-data/free1.csv')

df.head()
df.groupby(['country'])

df
df1=df.groupby('country').groups

df1
plt.clf()

df.groupby('country').size().plot(kind='bar')

plt.show()
df2=df.groupby('sex').groups

df2
plt.clf()

df.groupby('sex').size().plot(kind='bar')

plt.show()