import pandas as pd

import numpy as np

import datetime

import matplotlib.pyplot as plt
from pandas_datareader import data as pdr

import fix_yahoo_finance



aapl = pdr.get_data_yahoo('AAPL', 

                          start=datetime.datetime(2006, 10, 1), 

                          end=datetime.datetime(2012, 1, 1))

aapl.head()
import quandl 

aapl = quandl.get("WIKI/AAPL", start_date="2006-10-01", end_date="2012-01-01")

aapl.head()
aapl = pd.read_csv("https://s3.amazonaws.com/assets.datacamp.com/blog_assets/aapl.csv", header=0, index_col= 0, names=['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'], parse_dates=True)
aapl.index
aapl.columns
ts = aapl['Close'][-10:]
type(ts)
print(aapl.loc[pd.Timestamp('2006-11-01'):pd.Timestamp('2006-12-31')].head())
print(aapl.loc['2007'].head())
print(aapl.iloc[22:43])
print(aapl.iloc[[22,43], [0, 3]])
sample = aapl.sample(20)

print(sample)
monthly_aapl = aapl.resample('M').interpolate()

print(monthly_aapl.head(10))
aapl['diff'] = aapl.Open - aapl.Close
del aapl['diff']
import matplotlib.pyplot as plt
aapl['Close'].plot(grid=True)



plt.show()
daily_close = aapl[['Adj Close']]
daily_close = aapl[['Adj Close']]
daily_pct_c = daily_close.pct_change()
daily_pct_c.fillna(0, inplace=True)
print(daily_pct_c.head(10))
daily_log_returns = np.log(daily_close.pct_change()+1)



print(daily_log_returns.head(10))
monthly = aapl.resample('BM').apply(lambda x: x[-1])
monthly.pct_change().head(10)
quarter = aapl.resample("4M").mean()
quarter.pct_change()
daily_pct_c = daily_close / daily_close.shift(1) - 1



print(daily_pct_c.head(10))
import matplotlib.pyplot as plt



daily_pct_c.hist(bins=50)

plt.show()

print(daily_pct_c.describe())
cum_daily_return = (1 + daily_pct_c).cumprod()

print(cum_daily_return.tail()) 
cum_daily_return.plot(figsize=(12,8))

plt.show()
cum_monthly_return = cum_daily_return.resample("M").mean()

print(cum_monthly_return.tail())
from pandas_datareader import data as pdr

import fix_yahoo_finance



def get(tickers, startdate, enddate):

    def data(ticker):

        return (pdr.get_data_yahoo(ticker, start=startdate, end=enddate))

    datas = map (data, tickers)

    return(pd.concat(datas, keys=tickers, names=['Ticker', 'Date']))



tickers = ['AAPL', 'MSFT', 'IBM', 'GOOG']

all_data = get(tickers, datetime.datetime(2006, 10, 1), datetime.datetime(2012, 1, 1))

all_data.head()
#all_data = pd.read_csv("https://s3.amazonaws.com/assets.datacamp.com/blog_assets/all_stock_data.csv", index_col= [0,1], header=0, parse_dates=[1])
daily_close_px = all_data[['Adj Close']].reset_index().pivot('Date', 'Ticker', 'Adj Close')



# Calculate the daily percentage change for `daily_close_px`

daily_pct_change = daily_close_px.pct_change()



# Plot the distributions

daily_pct_change.hist(bins=50, sharex=True, figsize=(12,8))



# Show the resulting plot

plt.show()
pd.plotting.scatter_matrix(daily_pct_change, diagonal='kde', alpha=0.1,figsize=(12,12))

plt.show()
adj_close_px = aapl['Adj Close']

moving_avg = adj_close_px.rolling(window=40).mean()

moving_avg[-10:]
aapl['42'] = adj_close_px.rolling(window=40).mean()

aapl['252'] = adj_close_px.rolling(window=252).mean()

aapl[['Adj Close', '42', '252']].plot()



plt.show()
min_periods = 75 

vol = daily_pct_change.rolling(min_periods).std() * np.sqrt(min_periods) 

vol.plot(figsize=(10, 8))

plt.show()
import statsmodels.api as sm

from pandas import tseries

from pandas.core import datetools
all_adj_close = all_data[['Adj Close']]
all_returns = np.log(all_adj_close / all_adj_close.shift(1))
aapl_returns = all_returns.iloc[all_returns.index.get_level_values('Ticker') == 'AAPL']

aapl_returns.index = aapl_returns.index.droplevel('Ticker')
msft_returns = all_returns.iloc[all_returns.index.get_level_values('Ticker') == 'MSFT']

msft_returns.index = msft_returns.index.droplevel('Ticker')
return_data = pd.concat([aapl_returns, msft_returns], axis=1)[1:]

return_data.columns = ['AAPL', 'MSFT']
X = sm.add_constant(return_data['AAPL'])
model = sm.OLS(return_data['MSFT'],X).fit()

print(model.summary())
plt.plot(return_data['AAPL'], return_data['MSFT'], 'r.')



ax = plt.axis()

x = np.linspace(ax[0], ax[1] + 0.01)



plt.plot(x, model.params[0] + model.params[1] * x, 'b', lw=2)



plt.grid(True)

plt.axis('tight')

plt.xlabel('Apple Returns')

plt.ylabel('Microsoft returns')



plt.show()
return_data['MSFT'].rolling(window=252).corr(return_data['AAPL']).plot()

plt.show()
short_window = 40

long_window = 100
signals = pd.DataFrame(index=aapl.index)

signals['signal'] = 0.0
signals['short_mavg'] = aapl['Close'].rolling(window=short_window, min_periods=1, center=False).mean()
signals['long_mavg'] = aapl['Close'].rolling(window=long_window, min_periods=1, center=False).mean()
signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] 

                                            > signals['long_mavg'][short_window:], 1.0, 0.0)  
signals['positions'] = signals['signal'].diff()
fig = plt.figure()

ax1 = fig.add_subplot(111,  ylabel='Price in $')

aapl['Close'].plot(ax=ax1, color='r', lw=2.)

signals[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2.)

ax1.plot(signals.loc[signals.positions == 1.0].index, 

         signals.short_mavg[signals.positions == 1.0],

         '^', markersize=10, color='m')

ax1.plot(signals.loc[signals.positions == -1.0].index, 

         signals.short_mavg[signals.positions == -1.0],

         'v', markersize=10, color='k')

plt.show()
initial_capital= float(100000.0)
positions = pd.DataFrame(index=signals.index).fillna(0.0)
positions['AAPL'] = 100*signals['signal']   
portfolio = positions.multiply(aapl['Adj Close'], axis=0)
pos_diff = positions.diff()
portfolio['holdings'] = (positions.multiply(aapl['Adj Close'], axis=0)).sum(axis=1)
portfolio['cash'] = initial_capital - (pos_diff.multiply(aapl['Adj Close'], axis=0)).sum(axis=1).cumsum() 
portfolio['total'] = portfolio['cash'] + portfolio['holdings']
portfolio['returns'] = portfolio['total'].pct_change()
fig = plt.figure()



ax1 = fig.add_subplot(111, ylabel='Portfolio value in $')



portfolio['total'].plot(ax=ax1, lw=2.)



ax1.plot(portfolio.loc[signals.positions == 1.0].index, 

         portfolio.total[signals.positions == 1.0],

         '^', markersize=10, color='m')



ax1.plot(portfolio.loc[signals.positions == -1.0].index, 

         portfolio.total[signals.positions == -1.0],

         'v', markersize=10, color='k')



plt.show()
returns = portfolio['returns']
sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std())



print(sharpe_ratio)
window = 252
rolling_max = aapl['Adj Close'].rolling(window, min_periods=1).max()

daily_drawdown = aapl['Adj Close']/rolling_max - 1.0
max_daily_drawdown = daily_drawdown.rolling(window, min_periods=1).min()
daily_drawdown.plot()

max_daily_drawdown.plot()



plt.show()
days = (aapl.index[-1] - aapl.index[0]).days



cagr = ((((aapl['Adj Close'][-1]) / aapl['Adj Close'][1])) ** (365.0/days)) - 1



print(cagr)