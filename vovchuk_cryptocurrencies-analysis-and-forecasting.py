!pip install pytrends

!pip install Cryptory


import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import statsmodels.api as sm

from statsmodels.tsa.stattools import adfuller

import warnings

from scipy import stats





from cryptory import Cryptory



import os

from itertools import product

from datetime import datetime



from plotly import tools

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go



warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', 7)

all_coins_names = ['btc', 'eth', 'xrp', 'bch', 'ltc', 'dash', 'xmr', 'etc', 'zec', 

                        'doge', 'rdd', 'vtc', 'ftc', 'nmc', 'blk','nvc']
help(Cryptory)


# initialise object 

# pull data from start of 2011 to present day

my_cryptory = Cryptory(from_date = "2011-02-01", to_date = "2019-05-11" )



# get historical bitcoin prices from bitinfocharts

btc_price_data=my_cryptory.extract_bitinfocharts("btc")                                                 

btc_price_data = btc_price_data.set_index('date')

btc_price_data

# get historical bitcoin data from coinmarketcap

btc_marketcap = my_cryptory.extract_coinmarketcap("bitcoin")

btc_marketcap
plt.figure(figsize=(20,7))

ax = plt.subplot(111)    

plt.yticks(fontsize=14)    

plt.xticks(fontsize=14)

ax.spines["top"].set_visible(False)        

ax.spines["right"].set_visible(False)    

plt.grid(linewidth=0.2)

plt.title('Bitcoin Market Capitalization (2013-2018)', fontsize=18)

plt.plot(btc_marketcap.date,btc_marketcap.marketcap)

plt.show()
plt.figure(figsize=(20,7))

ax = plt.subplot(111)    

plt.yticks(fontsize=14)    

plt.xticks(fontsize=14)

ax.spines["top"].set_visible(False)        

ax.spines["right"].set_visible(False)

btc_price_data.btc_price.plot()

plt.grid(linewidth=0.2)

plt.title('Historical Bitcoin Prices (2013-2018)', fontsize=18)

plt.show()



plt.figure(figsize=(20,7))

ax = plt.subplot(111)    

plt.yticks(fontsize=14)    

plt.xticks(fontsize=14)

ax.spines["top"].set_visible(False)        

ax.spines["right"].set_visible(False)    

plt.grid(linewidth=0.2)

plt.title('Bitcoin Market Volume (2013-2018)', fontsize=18)

plt.plot(btc_marketcap.date,btc_marketcap.volume)

plt.show()

# Resampling to monthly frequency

df_month = btc_price_data.resample('M').mean()



# Resampling to annual frequency

df_year = btc_price_data.resample('A-DEC').mean()



# Resampling to quarterly frequency

df_Q = btc_price_data.resample('Q-DEC').mean()

fig = plt.figure(figsize=[20, 8])

plt.suptitle('Bitcoin exchanges, mean USD', fontsize=22)



plt.subplot(221)

plt.plot(btc_price_data.btc_price, '-', label='By Days')

plt.legend()



plt.subplot(222)

plt.plot(df_month.btc_price, '-', label='By Months')

plt.legend()



plt.subplot(223)

plt.plot(df_Q.btc_price, '-', label='By Quarters')

plt.legend()



plt.subplot(224)

plt.plot(df_year.btc_price, '-', label='By Years')

plt.legend()



# plt.tight_layout()

plt.show()
plt.figure(figsize=[20,8])

sm.tsa.seasonal_decompose(df_month.btc_price).plot()

plt.show()
def ADF(df):

    result = adfuller(df)

    print('ADF Statistic: %f' % result[0])

    print('p-value: %f' % result[1])

    print('Critical Values:')

    for key, value in result[4].items():

        print('\t%s: %.3f' % (key, value))
ADF(df_month.btc_price)
df_month['Price_Box'], lmbda = stats.boxcox(df_month.btc_price)

ADF(df_month.Price_Box)
df_month['Price_Box_Diff_12'] = df_month.Price_Box - df_month.Price_Box.shift(12)

ADF(df_month.Price_Box_Diff_12[12:])
df_month['Price_Box_Diff_13'] = df_month.Price_Box_Diff_12 - df_month.Price_Box_Diff_12.shift(1)

ADF(df_month.Price_Box_Diff_13[13:])
plt.figure(figsize=[20,8])

sm.tsa.seasonal_decompose(df_month.Price_Box_Diff_13[13:]).plot()

plt.show()
# Initial approximation of parameters using Autocorrelation and Partial Autocorrelation Plots

plt.figure(figsize=(15,7))

ax = plt.subplot(211)

sm.graphics.tsa.plot_acf(df_month.Price_Box_Diff_13[13:].values.squeeze(), lags=40, ax=ax)

ax = plt.subplot(212)

sm.graphics.tsa.plot_pacf(df_month.Price_Box_Diff_13[13:].values.squeeze(), lags=40, ax=ax)

plt.tight_layout()

plt.show()
# Initial approximation of parameters

Qs = range(0, 2)

qs = range(0, 3)

Ps = range(0, 3)

ps = range(0, 3)

D=1

d=1

parameters = product(ps, qs, Ps, Qs)

parameters_list = list(parameters)

len(parameters_list)



# Model Selection

results = []

best_aic = float("inf")

warnings.filterwarnings('ignore')

for p in parameters_list:

    try:

        model=sm.tsa.statespace.SARIMAX(df_month.Price_Box, order=(p[0], d, p[1]), 

                                        seasonal_order=(p[2], D, p[3], 12)).fit(disp=-1)

    except ValueError:

        print('wrong parameters:', p)

        continue

    aic = model.aic

    if aic < best_aic:

        best_model = model

        best_aic = aic

        best_param = p

    results.append([p, model.aic])
# Best Models

result_table = pd.DataFrame(results)

result_table.columns = ['parameters', 'aic']

print(result_table.sort_values(by = 'aic', ascending=True).head())

print(best_model.summary())
# STL-decomposition

plt.figure(figsize=(15,7))

plt.subplot(211)

best_model.resid[13:].plot()

plt.ylabel(u'Residuals')

ax = plt.subplot(212)

sm.graphics.tsa.plot_acf(best_model.resid[13:].values.squeeze(), lags=48, ax=ax)



ADF(best_model.resid[13:])



plt.tight_layout()

plt.show()
# Inverse Box-Cox Transformation

def invboxcox(y,lmbda):

   if lmbda == 0:

      return(np.exp(y))

   else:

      return(np.exp(np.log(lmbda*y+1)/lmbda))
# Prediction

df_month2 = df_month[['btc_price']]

date_list = [datetime(2019, 6, 30), datetime(2019, 7, 31), datetime(2019, 8, 31), datetime(2019, 9, 30), 

             datetime(2019, 10, 31), datetime(2019, 11, 30), datetime(2019, 12, 31), datetime(2020, 1, 31),

             datetime(2020, 1, 28)]

future = pd.DataFrame(index=date_list, columns= df_month.columns)

df_month2 = pd.concat([df_month2, future])

df_month2['forecast'] = invboxcox(best_model.predict(start=0, end=160), lmbda)

plt.figure(figsize=(15,7))

df_month2.btc_price.plot()

df_month2.forecast.plot(color='r', ls='--', label='Predicted Weighted_Price')

plt.legend()

plt.title('Bitcoin exchanges, by months')

plt.ylabel('mean USD')

plt.show()
def get_coins_price(coins,start_date, end_date):

    cryptory = Cryptory(from_date = start_date, to_date = end_date)

    all_coins_df = cryptory.extract_bitinfocharts(coins[0])

    for coin in coins[1:]:

        all_coins_df = all_coins_df.merge(cryptory.extract_bitinfocharts(coin), on="date", how="left")

    return all_coins_df





def draw_pearson_correlation(start_date, end_date):

    all_coins_df = get_coins_price(all_coins_names,start_date, end_date)

    corr = all_coins_df.iloc[:,1:].pct_change().corr(method='pearson')

    corr = corr.dropna(axis=0, how='all').dropna(axis=1, how='all').round(2)

    sns.set(font_scale=1)

    sns.heatmap(corr, 

                xticklabels=[col.replace("_price", "") for col in corr.columns.values],

                yticklabels=[col.replace("_price", "") for col in corr.columns.values],

                annot=True, linewidths=.5,

                vmin=0, vmax=1)
fig = plt.figure(figsize=[22, 20])

plt.tight_layout()

i = 221

for year in range(2016,2019):

    sub=plt.subplot(i)

    sub.set_title("Correlation of crypto in "+str(year), fontsize=22)

    draw_pearson_correlation(datetime(year,1,1),datetime(year,11,3))

    i+=1

plt.show()
btc_google = my_cryptory.get_google_trends(kw_list=['bitcoin'])

btc_google.index = btc_google.date

btc_google


k = btc_google.resample('M').mean()



plt.figure(figsize=(20,7))

ax = plt.subplot(111)    

plt.yticks(fontsize=14)    

plt.xticks(fontsize=14)

ax.spines["top"].set_visible(False)        

ax.spines["right"].set_visible(False)    

plt.grid(linewidth=0.2)

plt.title('BTC x USD (Bitfinex)')

plt.plot(df_month.btc_price.apply(np.log))

plt.plot(k.bitcoin.apply(np.log))



plt.show()
selected = ['btc','eth','ltc','dash']

data = get_coins_price(selected,datetime(2016,1,1), datetime(2016,12,31))

data
clean = data.set_index('date')

table = clean.sort_values(by = 'date',ascending = True).dropna()



# calculate daily and annual returns of crypto

returns_daily = table.pct_change()

returns_annual = returns_daily.mean() * 365



# get daily and covariance of returns of crypto

cov_daily = returns_daily.cov()

cov_annual = cov_daily * 365



# returns, volatility and weights of portfolios

port_returns = []

port_volatility = []

sharpe_ratio = []

stock_weights = []



num_assets = len(selected)

num_portfolios = 50000



np.random.seed(101)



for single_portfolio in range(num_portfolios):

    weights = np.random.random(num_assets)

    weights /= np.sum(weights)

    returns = np.dot(weights, returns_annual)

    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))

    sharpe = returns / volatility

    sharpe_ratio.append(sharpe)

    port_returns.append(returns)

    port_volatility.append(volatility)

    stock_weights.append(weights)



# a dict with Returns and Risk values of portfolio

portfolio = {'Returns': port_returns,

             'Volatility': port_volatility,

             'Sharpe Ratio': sharpe_ratio}



for counter,symbol in enumerate(selected):

    portfolio[symbol+' Weight'] = [Weight[counter] for Weight in stock_weights]



# create dataframe

df = pd.DataFrame(portfolio)



# change labels

column_order = ['Returns', 'Volatility', 'Sharpe Ratio'] + [stock+' Weight' for stock in selected]



# reorder columns

df = df[column_order]
plt.style.use('seaborn-dark')

df.plot.scatter(x='Volatility', y='Returns', c='Sharpe Ratio',

                cmap='magma', edgecolors='black', figsize=(10, 10), grid=True)

plt.xlabel('Volatility (Std. Deviation)')

plt.ylabel('Expected Returns')

plt.title('Efficient Frontier')

plt.show()
# find min Volatility & max sharpe values

min_volatility = df['Volatility'].min()

max_sharpe = df['Sharpe Ratio'].max()



sharpe_portfolio = df.loc[df['Sharpe Ratio'] == max_sharpe]

min_variance_port = df.loc[df['Volatility'] == min_volatility]

plt.style.use('seaborn-dark')

df.plot.scatter(x='Volatility', y='Returns', c='Sharpe Ratio',

                cmap='magma', edgecolors='black', figsize=(10, 8), grid=True)

plt.scatter(x=sharpe_portfolio['Volatility'], y=sharpe_portfolio['Returns'], c='red', marker='D', s=100)

plt.scatter(x=min_variance_port['Volatility'], y=min_variance_port['Returns'], c='blue', marker='D', s=100 )

plt.xlabel('Volatility (Std. Deviation)')

plt.ylabel('Expected Returns')

plt.title('Efficient Frontier')

plt.show()


min_variance_port.T
sharpe_portfolio.T