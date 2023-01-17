#Libraries

import pandas as pd

import numpy as np

from pandas_datareader import data as wb



import seaborn as sns

from matplotlib import pyplot as plt

%matplotlib inline



import warnings

warnings.filterwarnings("ignore")
#Download data

TSLA = wb.DataReader('TSLA', data_source="yahoo",start = "2018-1-1",end='2018-12-31')

TSLA.head()
GOOGL = wb.DataReader('GOOGL', data_source="yahoo",start = '2018-1-1',end='2018-12-31')

GOOGL.head()
AMZN = wb.DataReader('AMZN', data_source="yahoo",start = '2018-1-1',end='2018-12-31')

AMZN.head()
AAPL = wb.DataReader('AAPL', data_source="yahoo",start = '2018-1-1',end='2018-12-31')

AAPL.head()
#Individual stock analysis 

#Simple RR



TSLA['Simple Return'] = (TSLA["Adj Close"]/TSLA["Adj Close"].shift(1))-1

TSLA.head()
#Visualization of the rate of return data

TSLA['Simple Return'].plot(figsize=(12,8))
#A similar picture can be presented as a histogram of the daily return

fig = plt.figure(figsize=(12,8))

sns.distplot(TSLA['Simple Return'].dropna(),bins=100)
!pip install mpl-finance
#Creating a candlestick plot for share prices:

import mpl_finance

from mpl_finance import candlestick_ohlc

import datetime

import matplotlib.dates as mdates





TSLA_candle = TSLA[['Open', 'High', 'Low', 'Close']]

TSLA_candle.reset_index(inplace=True)

TSLA_candle['Date'] = TSLA_candle['Date'].map(mdates.date2num)



fig = plt.figure(figsize=(12,8))

ax = plt.subplot()

candlestick_ohlc(ax, TSLA_candle.values, width=5, colorup='g', colordown='r')

ax.xaxis_date()

ax.grid(True)

plt.show()
#Compute return rates for the other shares in the portfolio

GOOGL['Simple Return'] = (GOOGL["Adj Close"]/GOOGL["Adj Close"].shift(1))-1

GOOGL.head()
AMZN['Simple Return'] = (AMZN["Adj Close"]/AMZN["Adj Close"].shift(1))-1

AMZN.head()
AAPL['Simple Return'] = (AAPL["Adj Close"]/AAPL["Adj Close"].shift(1))-1

AAPL.head()
#Daily return plot

f, ax = plt.subplots(2,2, figsize = (11,8))

k1 = sns.lineplot(y = TSLA["Simple Return"], x = TSLA.index.get_level_values(0), ax = ax[0,0])

k2 = sns.lineplot(y = GOOGL["Simple Return"], x = GOOGL.index.get_level_values(0), ax = ax[0,1])

k3 = sns.lineplot(y = AMZN["Simple Return"], x = AMZN.index.get_level_values(0), ax = ax[1,0])

k4 = sns.lineplot(y = AAPL["Simple Return"], x = AAPL.index.get_level_values(0), ax = ax[1,1])



graphs = [k1,k2,k3,k4]



for i in graphs:

    i.tick_params(axis='x', labelsize=8,labelrotation=90)



k1.title.set_text('Tesla')

k2.title.set_text('Google')

k3.title.set_text('Amazon')

k4.title.set_text('Apple')

plt.tight_layout()
#Calculate annualised average return

TSLA_avg_return = round(TSLA['Simple Return'].mean()*250, 2)*100

print(str(TSLA_avg_return) + '%')
#Calculte annualized mean log return 



TSLA['Log Return'] = np.log(TSLA['Adj Close']/TSLA['Adj Close'].shift(1))
#Calculate moving average

ma_days = [20, 50, 200]



for ma in ma_days:

    new_columns = '%s day MA' %(str(ma))

    TSLA[new_columns] = TSLA['Adj Close'].rolling(ma).mean()

TSLA[['Adj Close','20 day MA', '50 day MA', '200 day MA']].plot(subplots=False, figsize=(12,8), title="Tesla price/moving averages")
#join DFrames for portfolio analysis

ma_data = pd.concat([TSLA['Adj Close'], GOOGL['Adj Close'], AMZN['Adj Close'],AAPL['Adj Close']], axis=1).dropna()

ma_data.columns = ['TSLA','GOOGL','AMZN','AAPL']

ma_data.head()
#Moving averages for multiple shares



    

#Create a dataset with all the moving average values    

m = [20,50,200]

ma_20 = ma_data.rolling(20).mean()

ma_50 = ma_data.rolling(50).mean()

ma_200 = ma_data.rolling(200).mean()

ma_concat = pd.concat([ma_20,ma_50,ma_200], axis=1)





#Create a subplot for the moving averages of the shares

list_ma = ['TSLA','GOOGL','AMZN','AAPL']



f, ax = plt.subplots(2, 2, figsize = (11,8))

dict_ma = {'TSLA':[0,0],'GOOGL':[0,1], 'AMZN':[1,0],'AAPL':[1,1]}

for k, v in dict_ma.items():

    g1 = ma_concat[k].plot(ax=ax[v[0], v[1]], label=k, legend=False)

    g1.set_title(k)

    



L=plt.legend()

L.get_texts()[0].set_text('20 day MA')

L.get_texts()[1].set_text('50 day MA')

L.get_texts()[2].set_text('200 day MA')

plt.tight_layout()
#Portfolio analysis

from pandas import DataFrame

tickers = ['TSLA', 'GOOGL', 'AMZN','AAPL']

my_portfolio = pd.DataFrame()



#Create the same DF as above directly from the data source

for t in tickers:

    my_portfolio[t] = wb.DataReader(t, data_source="yahoo",start = "2018-1-1", end='2018-12-31')['Adj Close']



my_portfolio.head()
#Index data

NASDAQComp = wb.DataReader('^IXIC', data_source="yahoo",start = '2018-1-1', end='2018-12-31')
#Adding the index to the DF

tickersI = ['TSLA', 'GOOGL', 'AMZN','AAPL', '^IXIC']

my_portfolioI = pd.DataFrame()



for t in tickersI:

    my_portfolioI[t] = wb.DataReader(t, data_source="yahoo",start = '2018-1-1', end='2018-12-31')['Adj Close']



my_portfolioI.head()
index_return = (my_portfolioI['^IXIC']/my_portfolioI['^IXIC'].shift(1))-1

index_return.head()
#To see how the stock prices changed during the period

plot_portfolio = (my_portfolio/my_portfolio.iloc[0]*100).plot(figsize = (12,8))

plot_portfolio
#To see how the securities performed against the index during the period

f, ax = plt.subplots(figsize = (12,8))

(my_portfolioI.loc[:, my_portfolioI.columns != "^IXIC"]/my_portfolioI.loc[:, my_portfolioI.columns != "^IXIC"].iloc[0]*100).plot(ax=ax, linewidth=1.5)

(my_portfolioI[['^IXIC']]/my_portfolioI[['^IXIC']].iloc[0]*100).plot(linewidth=3.5, ax=ax)
#Portfolio RR - the simple rate of return for each share in the portfolio for each day

portfolio_returns = (my_portfolio/my_portfolio.shift(1))-1

portfolio_returns.head()
#Portfolio RR - the simple RR for each share in the portfolio for each day and weighted for the portfolio

weights = np.array([0.30,0.20,0.25,0.15]) #an example for the weight of each share in the portfolio

ret = my_portfolio.pct_change()

ret['Portfolio RR'] = np.dot(ret,weights)

ret.head()
#To plot the relationships between each stock pair in the portfolio

ret_fig = sns.pairplot(ret.dropna())
#To customise the above graph

ret_fig = sns.PairGrid(ret.dropna(),height=3)

ret_fig.map_diag(plt.hist,bins=30)

ret_fig.map_upper(sns.kdeplot, shade=True, shade_lowest=False)

ret_fig.map_lower(plt.scatter,color='green',alpha=0.1)
#Annualised returns for each share

annual_returns = portfolio_returns.mean()*250

annual_returns
#Portfolio return

print(str(round(np.dot(annual_returns, weights),4)*100) + '%')
#Calculate share log returns to find portfolio covariance and correlation

my_portfolio_log = np.log(my_portfolio/my_portfolio.shift(1))

my_portfolio_log.head()
#Portfolio returns covariance

cov_matrix = my_portfolio_log.cov()*250

cov_matrix
#Share returns correlation

corr_matrix = my_portfolio_log.corr()

corr_matrix
#Correlation plot of the returns



plt.figure(figsize=(12,8))

mask = np.zeros_like(corr_matrix)

mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):

    ax = sns.heatmap(corr_matrix, mask=mask, annot=True,vmin=-1,cmap='coolwarm', vmax=.4)
#Portfolio risk

portfolio_variance = np.dot(weights.T, np.dot(my_portfolio_log.cov()*250, weights))

print("Portfolio variance: " + str(round(portfolio_variance, 4)))



portfolio_volatility = (np.dot(weights.T, np.dot(my_portfolio_log.cov()*250,weights)))**0.5

print("Portfolio standard deviation: " + str(round(portfolio_volatility, 4)))

#Risk analysis: find the VaR

plt.figure(figsize=(12,8))

sns.distplot(ret['Portfolio RR'].dropna(),bins=100)

#Using the empirical quantile method to define the maximum daily loss of our portfolio with 95% confidence. 

#In this case our lowest return (i.e. loss) will not exceed -2.89%. 

portfolio_var = ret['Portfolio RR'].dropna().quantile(0.05)

print(str(round(portfolio_var*100,2)) + '%')

#Running a regresson between the rates of return of the index and our portfolio

import statsmodels.api as sm



x = index_return.dropna()

y = ret['Portfolio RR'].dropna()



results = sm.OLS(y,sm.add_constant(x)).fit()

print(results.summary())





plt.figure(figsize=(12,8))

plt.scatter(x,y)



x_plot = np.linspace(0,1,100)

plt.plot(x, results.fittedvalues)



plt.show()



#or calculate just the main values in OLS

from scipy import stats

stats = stats.linregress(x,y)

print(stats)

#Monte Carlo simulation for the portfolio rate of return

days = 365

dt = 1./days

mu = ret['Portfolio RR'].mean()

sigma = ret['Portfolio RR'].std()

mu, sigma

def monte_carlo_portfolio(start_price,days,mu,sigma):

    

    # Define a price array

    RR = np.zeros(days)

    RR[0] = start_RR

    # Schok and Drift

    shock = np.zeros(days)

    drift = np.zeros(days)

    

    # Run price array for number of days

    for x in range(1,days):

        

        # Calculate Schock

        shock[x] = np.random.normal(loc=mu * dt, scale=sigma * np.sqrt(dt))

        # Calculate Drift

        drift[x] = mu * dt

        # Calculate Price

        RR[x] = RR[x-1] + (RR[x-1] * (drift[x] + shock[x]))

        

    return RR

start_RR = 0.032741



plt.figure(figsize=(12,8))

for run in range(1000):

    plt.plot(monte_carlo_portfolio(start_RR,days,mu,sigma))



    

plt.xlabel("Days")

plt.ylabel("Price")  

plt.title('Portfolio Monte Carlo Analysis')

#To make a histogram of 1000 simulations

runs = 1000



# Create an empty matrix to hold the data and then populate it with simulation results

simulations = np.zeros(runs)



for run in range(runs):    

    # Set the simulation data point as the last rate of return for that run

    simulations[run] = monte_carlo_portfolio(start_RR,days,mu,sigma)[days-1]

# We can use 1% empirical qunatile to find the lowest expected return, which means that 99% of the values should fall 

#between here and the end RR

q = np.percentile(simulations, 1)

    

# Plot the distribution of the end RR values

fig = plt.figure(figsize=(12,8))

plt.hist(simulations,bins=200)



# Using plt.figtext to fill in some additional information onto the plot



# Starting RR



plt.figtext(0.7, 0.8, s="Start RR: %.4f" %start_RR)

# Mean ending price

#plt.figtext(0.7, 0.7, "Mean RR: %.2f" % simulations.mean())



# Value at risk within 99% confidence interval

plt.figtext(0.7, 0.7, "VaR(0.99): %.4f" % (start_RR - q,))



# To see the 1% quantile

plt.figtext(0.15, 0.6, "q(0.99): %.4f" % q)



# Plot a line at the 1% quantile result

plt.axvline(x=q, linewidth=4, color='r')
#Portfolio sharpe ratio, assuming a 0 risk free rate. This enables investors to

#compare portfolios (or stocks) in terms of risk-return performance.



sharpe_ratio = ret['Portfolio RR'].mean()/ret['Portfolio RR'].std()



#Annualise the ratio

sharpe_ratio_annualised = sharpe_ratio*(250**0.5)

round(sharpe_ratio_annualised,2)