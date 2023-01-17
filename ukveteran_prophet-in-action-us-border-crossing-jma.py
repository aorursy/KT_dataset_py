# Importing neccesary packages.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import warnings

#

import statsmodels.api as sm
from pylab import rcParams
import scipy.stats as ss

warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')
data = pd.read_csv('/kaggle/input/us-border-crossing-data/Border_Crossing_Entry_Data.csv')
data.shape
data.sample(5)
data['Date'] = pd.to_datetime(data['Date'])
data.rename(columns={'Value': 'Total Entries'}, inplace=True)
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
from fbprophet import Prophet
data.head()
df1=data.rename(columns={"Date": "ds", "Total Entries": "y"})
df1
m = Prophet()
m.fit(df1)
future = m.make_future_dataframe(periods=365)
future.tail()
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)