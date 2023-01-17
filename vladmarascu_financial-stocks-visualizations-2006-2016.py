from pandas_datareader import data, wb

import pandas as pd

import numpy as np

import seaborn as sns

import datetime

%matplotlib inline
start = datetime.datetime(2006, 1, 1)

end = datetime.datetime(2016, 1, 1)
# Bank of America

BAC = data.DataReader("BAC", 'yahoo', start, end)



# CitiGroup

C = data.DataReader("C", 'yahoo', start, end)



# Goldman Sachs

GS = data.DataReader("GS", 'yahoo', start, end)



# JPMorgan Chase

JPM = data.DataReader("JPM", 'yahoo', start, end)



# Morgan Stanley

MS = data.DataReader("MS", 'yahoo', start, end)



# Wells Fargo

WFC = data.DataReader("WFC", 'yahoo', start, end)
BAC.head(3)
# ALTERNATIVE TO CONCAT (METHOD 1) -- GROUPED BY OPEN, CLOSE ETC

df = data.DataReader(['BAC', 'C', 'GS', 'JPM', 'MS', 'WFC'],'yahoo', start, end)

df.head(3)
tickers = ['BAC', 'C', 'GS', 'JPM', 'MS', 'WFC']
bank_stocks = pd.concat([BAC, C, GS, JPM, MS, WFC],axis=1,keys=tickers) # CONCAT ON COLUMNS AXIS=1
bank_stocks.columns.names = ['Bank Ticker','Stock Info'] # 2 LEVELS

bank_stocks.head(3)
bank_stocks=pd.read_pickle('../input/databanks/all_banks') # overwrite the DF with the file containing this (will be uploaded on Kaggle as data source)

bank_stocks.head(3) # GROUPED BY BANK NAMES
bank_stocks.xs(key='Close',axis=1,level='Stock Info').max()
# TO RETRIEVE ONE COLUMN ONLY

bank_stocks['BAC']['Close'] # BAC is a tick in tickers
# USING DF

df['Close'].max()
returns=pd.DataFrame()
for tick in tickers:

    returns[tick+' Return']=bank_stocks[tick]['Close'].pct_change()

    

returns.head() # first values are NaN
#sns.set_style('whitegrid')

sns.pairplot(returns[1:])
returns.idxmin()
returns.idxmax()
returns.std()
returns.loc['2015-01-01':'2016-01-01'].std() # for 2015 only (std)
sns.distplot(returns.loc['2015-01-01':'2016-01-01']['MS Return'],bins=50)
sns.distplot(returns.loc['2008-01-01':'2009-01-01']['C Return'],color='red',bins=50)
import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline
bank_stocks.xs(key='Close',axis=1,level='Stock Info').plot(figsize=(15,5))
# SAME PLOT USING PLOTLY



import plotly

import cufflinks as cf

cf.go_offline()



bank_stocks.xs(key='Close',axis=1,level='Stock Info').iplot()
plt.figure(figsize=(12,6))

BAC['Close'].loc['2008-01-01':'2009-01-01'].rolling(window=30).mean().plot(label='30 Day Average')

BAC['Close'].loc['2008-01-01':'2009-01-01'].plot(label='BAC Closing stock price')

plt.legend()
#bank_stocks.xs(key='Close',axis=1,level='Stock Info').corr() # CORR MATRIX

sns.heatmap(bank_stocks.xs(key='Close',axis=1,level='Stock Info').corr(),annot=True,cmap='viridis') # PLOT CORR ON HEATMAP
sns.clustermap(bank_stocks.xs(key='Close',axis=1,level='Stock Info').corr(),annot=True,cmap='viridis')
bac15=BAC[['Open', 'High', 'Low', 'Close']].loc['2015-01-01':'2016-01-01'] # SELECT BAC stocks prices in 2015 (time-series)

bac15.iplot(kind='candle')
ms15=MS['Close'].loc['2015-01-01':'2016-01-01']

ms15.ta_plot(study='sma',periods=[13,21,55],title='Simple Moving Averages')
bac15_bb=BAC['Close'].loc['2015-01-01':'2016-01-01']

bac15_bb.ta_plot(study='boll')