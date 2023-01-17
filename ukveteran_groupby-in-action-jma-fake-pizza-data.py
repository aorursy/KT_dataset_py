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
df = pd.read_csv('../input/fake-pizza-data/Fake Pizza Data.csv')

df.head()
df1=df.groupby('Neighborhood').groups

df1
df1=df.groupby('HeatSource').groups

df1
plt.clf()

df.groupby('HeatSource').size().plot(kind='bar')

plt.show()
plt.clf()

df.groupby('Neighborhood').size().plot(kind='bar')

plt.show()