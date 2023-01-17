# data analysis and wrangling

import pandas as pd

from pandas_datareader import data

import numpy as np

import random as rnd

import datetime



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

sns.set_style('whitegrid')

%matplotlib inline



# plotly

import plotly

import plotly.graph_objects as go

import plotly.express as px

from plotly.subplots import make_subplots

import cufflinks as cf

cf.go_offline()
# set the start date and end date.

start = datetime.datetime(2006,1,1)

end = datetime.datetime(2020,1,1)



# set each bank to be a separate dataframe

BAC = data.DataReader("BAC", 'yahoo', start, end)

C = data.DataReader("C", 'yahoo', start, end)

GS = data.DataReader("GS", 'yahoo', start, end)

JPM = data.DataReader("JPM", 'yahoo', start, end)

MS = data.DataReader("MS", 'yahoo', start, end)

WFC = data.DataReader("WFC", 'yahoo', start, end)



# in case you cannot access the data from yahoo finance

# BAC = pd.read_csv('../input/banks-historical-stock-price/BAC.csv')

# C = pd.read_csv('../input/banks-historical-stock-price/C.csv')

# GS = pd.read_csv('../input/banks-historical-stock-price/GS.csv')

# JPM = pd.read_csv('../input/banks-historical-stock-price/JPM.csv')

# MS = pd.read_csv('../input/banks-historical-stock-price/MS.csv')

# WFC = pd.read_csv('../input/banks-historical-stock-price/WFC.csv')
# preview one dataframe

BAC.head()
tickers = ['BAC', 'C', 'GS', 'JPM', 'MS', 'WFC']
bank_stocks = pd.concat([BAC, C, GS, JPM, MS, WFC], axis=1, keys=tickers)

# bank_stocks
bank_stocks.columns.names = ['Bank Ticker','Stock Info']

bank_stocks.head()
bank_stocks.xs('Close', axis=1, level='Stock Info').max()
C['Close'].idxmax()
# empty dataframe

returns = pd.DataFrame()
for tick in tickers:

    returns[tick + ' Return'] = bank_stocks[tick]['Close'].pct_change()



returns.head(3)
sns.pairplot(returns[1:])
# Biggest single day losses

returns.idxmin()
# Standard Deviation 

returns.std().plot(kind='bar', color='Green')

plt.ylabel('Standard Deviation')

plt.title('Banks Standard Deviation', fontsize=15)

sns.despine()
# 2015 Standard Deviation

returns.loc['2015-01-01':'2015-12-31'].std().plot(kind='bar', color='Green')

plt.ylabel('Standard Deviation')

plt.title('2015 Banks Standard Deviation', fontsize=15)

sns.despine()
plt.figure(figsize=(10,8))

ax = sns.distplot(returns.loc['2015-01-01':'2015-12-31']['MS Return'], color='green', bins=50)

ax.set_title('2015 Morgan Stanley Returns')

ax.set_xlabel('Returns')

ax.set_ylabel('Numbers of Returns')
plt.figure(figsize=(10,8))

ax = sns.distplot(returns.loc['2008-01-01':'2008-12-31']['C Return'], color='red', bins=50)

ax.set_title('2008 CitiGroup Returns')

ax.set_xlabel('Returns')

ax.set_ylabel('Numbers of Returns')
plt.figure(figsize=(7,6))

plt.title('Pearson Correlation Matrix',fontsize=15)

sns.heatmap(bank_stocks.xs(key='Close', axis=1, level='Stock Info').corr(), annot=True, cmap='GnBu',

            linewidths=0.25, linecolor='w', cbar_kws={"shrink": .7})
sns.clustermap(bank_stocks.xs(key='Close', axis=1, level='Stock Info').corr(), annot=True, cmap='coolwarm')
# Option N°1

for tick in tickers:

    bank_stocks[tick]['Close'].plot(label=tick, figsize=(12,6))

plt.legend()
# Option N°2

bank_stocks.xs('Close', axis=1, level='Stock Info').plot(figsize=(12,6))
# Option N°3

bank_stocks.xs('Close', axis=1, level='Stock Info').iplot(xTitle='Date', yTitle='Close Price', title='Cufflinks - Close Price for Each Bank')
fig = make_subplots(rows=3, cols=2)



trace0 = go.Histogram(x=returns.loc['2018-01-01':'2019-12-31']['BAC Return'], nbinsx=50, name="BAC")

trace1 = go.Histogram(x=returns.loc['2018-01-01':'2019-12-31']['C Return'], nbinsx=50, name="C")

trace2 = go.Histogram(x=returns.loc['2018-01-01':'2019-12-31']['GS Return'], nbinsx=50, name="GS")

trace3 = go.Histogram(x=returns.loc['2018-01-01':'2019-12-31']['JPM Return'], nbinsx=50, name="JPM")

trace4 = go.Histogram(x=returns.loc['2018-01-01':'2019-12-31']['MS Return'], nbinsx=50, name="MS")

trace5 = go.Histogram(x=returns.loc['2018-01-01':'2019-12-31']['WFC Return'], nbinsx=50, name="WFC")



fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)

fig.append_trace(trace2, 2, 1)

fig.append_trace(trace3, 2, 2)

fig.append_trace(trace4, 3, 1)

fig.append_trace(trace5, 3, 2)



fig.update_layout(title_text='Banks Returns (2018 - 2019)')



fig.show()
plt.figure(figsize=(12,6))

BAC['Close'].loc['2008-01-01':'2008-12-31'].rolling(window=30).mean().plot(color='blue', label='30 Day Moving Average')

BAC['Close'].loc['2008-01-01':'2008-12-31'].plot(color='green', label='BAC Close Price')

plt.ylabel('Close Price')

plt.xlabel('')

plt.title('Bank Of America Moving Average')

plt.legend()

BAC['Close'].loc['2008-01-01':'2008-12-31'].iplot(fill=True,colors=['green'])
MS.loc['2015-01-01':'2016-01-01']['Close'].ta_plot(study='sma', periods=[13,21,55])
# Bank of America Candlestick Chart

fig = go.Figure(data=[go.Candlestick(x=BAC.index,

                open=BAC['Open'],

                high=BAC['High'],

                low=BAC['Low'], 

                close=BAC['Close'])

                ])



fig.update_layout(

    title='Bank of Amercia Stock Price',

    yaxis_title='BAC Stock',

    shapes = [dict(

        x0='2009-01-20', x1='2009-01-20', y0=0, y1=1, xref='x', yref='paper', line_width=2),

             dict(

        x0='2007-12-01', x1='2007-12-01', y0=0, y1=1, xref='x', yref='paper', line_width=2)],

    annotations=[dict(

        x='2009-01-20', y=0.95, xref='x', yref='paper',

        showarrow=False, xanchor='left', text='President Obama Took Office'), 

                 dict(

        x='2007-12-01', y=0.1, xref='x', yref='paper',

        showarrow=False, xanchor='right', text='Subprime Mortgage Crisis')]

)



fig.show()
# Bank of America Candlestick Chart

BAC[['Open', 'High', 'Close', 'Low']].loc['2015-01-01':'2016-01-01'].iplot(kind='candle', 

                                                                           title='Bank of Amercia Stock Price', 

                                                                           yaxis_title='BAC Stock')
BAC.loc['2015-01-01':'2016-01-01']['Close'].ta_plot(study='boll',periods=14, title='Bollinger Bands')
fig = px.area(bank_stocks.xs(key='Close', axis=1, level='Stock Info'), facet_col="Bank Ticker", facet_col_wrap=2)

fig.show()
# Citigroup OHLC Chart

fig = go.Figure(data=go.Ohlc(x=C.index,

                    open=C['Open'],

                    high=C['High'],

                    low=C['Low'],

                    close=C['Close']))



fig.update_layout(

    title='Citigroup Stock Price',

    yaxis_title='C Stock',

    shapes = [dict(

        x0='2009-01-20', x1='2009-01-20', y0=0, y1=1, xref='x', yref='paper', line_width=2),

             dict(

        x0='2007-12-01', x1='2007-12-01', y0=0, y1=1, xref='x', yref='paper', line_width=2)],

    annotations=[dict(

        x='2009-01-20', y=0.95, xref='x', yref='paper',

        showarrow=False, xanchor='left', text='President Obama Took Office'), 

                 dict(

        x='2007-12-01', y=0.1, xref='x', yref='paper',

        showarrow=False, xanchor='right', text='Subprime Mortgage Crisis')]

)



fig.show()