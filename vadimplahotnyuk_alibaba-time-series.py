#importing all of the neccesary libraries



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from statsmodels.tsa.arima_model import ARIMA

import pandas_datareader

import datetime



import pandas_datareader.data as web
#creating a start and an end dates:

start = datetime.datetime(2017,1,1)

end = datetime.datetime.today()
#Alibaba

alibaba = web.DataReader('BABA', 'yahoo', start, end)

alibaba.head()


alibaba['Open'].plot(label = 'Alibaba')

plt.legend(loc = 'best');


alibaba['Volume'].plot(label = 'Alibaba')

plt.legend();


alibaba['Total Traded'] = alibaba['Open']*alibaba['Volume']

alibaba['Total Traded'].plot(figsize = (16,8), label = 'Alibaba')

plt.legend(loc = 'best');


alibaba['Avg'] = alibaba[['High', 'Low']].mean(axis=1)

alibaba['Total Traded New'] = alibaba['Avg']*alibaba['Volume']

alibaba['Total Traded New'].plot(figsize = (16,8), label = 'Alibaba')

plt.legend(loc = 'best');




alibaba['MA50'] = alibaba['Open'].rolling(50).mean()

alibaba['MA100'] = alibaba['Open'].rolling(100).mean()
alibaba['Returns'] = alibaba['Close'].pct_change(1)
alibaba['Returns'].hist(bins = 50);


alibaba['Returns'].hist(bins = 100, figsize=(16,8), label = 'Alibaba', alpha=0.4)

plt.legend();


alibaba['Returns'].plot(kind='kde', label = 'Alibaba', figsize=(10,8))

plt.legend();
box_df = pd.concat([ alibaba['Returns']], axis = 1)

box_df.columns = [ 'Alibaba Ret']

box_df.plot(kind='box', figsize = (8,10));
model = ARIMA(alibaba['Total Traded New'], order=(2,0,0)).fit()

y2 = model.predict(100)



y2.plot()