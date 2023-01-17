import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import statsmodels.tsa.api as smt

import statsmodels.api as sm

import statsmodels.stats as sms

import scipy.stats as scs

import pylab

from scipy import stats, integrate

import warnings

import seaborn as sns

warnings.filterwarnings("ignore")

import os

#print(os.listdir("../input"))

dat = pd.read_csv("../input/Poloniex_BTCETH_OrderBookFlow_Sample.csv")

print(dat.shape)

dat.head()
#select only trades

ticks = dat[dat['typeOT'] == 't']

ticks['tim1'] = pd.to_datetime(ticks['timeDateOTh'], format='%Y-%m-%d %H:%M:%S.%f')



#create copy, just in case

ticks1 = ticks

ticks1 = ticks1.set_index('tim1')



#create three timeframes, 1minute, 5minute and 30minute

t5 = pd.DataFrame()

#resample rate and amount as last values and sum respectivly on 5 minute interval

t5['5Tprice'] = ticks1.rate.resample('5T').last()

t5['5Tvol'] = ticks1.amount.resample('5T').sum()

t5.head()
#plot price, and moving averages

t5['ma10'] = t5['5Tprice'].rolling(10).mean()

t5['ma100'] = t5['5Tprice'].rolling(100).mean()

t5['ma200'] = t5['5Tprice'].rolling(200).mean()

ticks.fillna(0.076638)

plt.plot(t5['5Tprice'])

plt.plot(t5['ma10'])

plt.plot(t5['ma100'])

plt.plot(t5['ma200'])

plt.title("Moving Averages")

plt.legend()

plt.show()
#Split date and time

ticks[['Date','Time']] = ticks['timeDateOTh'].str.split(' ', 1,expand=True)

ticks['Time'] = pd.to_datetime(ticks['Time'], format='%H:%M:%S.%f')

#add column Hour of the day

#ticks['m'] = ticks['Time'].dt.minute

ticks['H'] = ticks['Time'].dt.hour

#ticks['D'] = ticks['Date'].dt.day

#ticks['M'] = ticks['Date'].dt.month

#ticks['Y'] = ticks['Date'].dt.year

# sum volume per hour of the day

volumePerHour = ticks.groupby(['H'])['amount'].sum().reset_index()

#Clearly people sleep during night :)

#sns.violinplot(x=volumePerHour['H'], y=volumePerHour['amount']);



plt.plot(volumePerHour['H'],volumePerHour['amount'])

plt.title("Volume per 24 hour")
#returns

t5['ret'] = t5['5Tprice'].diff()

#logReturns

t5['logRet'] = np.log(t5['5Tprice']).diff()

#fillNa

t5 = t5.fillna(0)

#select first 100, just to check

#ticks1 = ticks.head(100)

#plot

plt.plot(t5['logRet'])

plt.title("ETH/BTC ticks return")

plt.show()
#cumulative returns

t5['cumRet'] = (1 + t5['logRet']).cumprod()

plt.plot(t5['cumRet'])

plt.title("Cumulative return")

plt.show()
# get distribution of returns, and fit a chosen distribution to see fit

#fig, axs = plt.subplots(ncols=2)

sns.distplot(t5['logRet'], kde=True, fit=stats.norm)

#sns.distplot(t5['logRet'],

#             hist_kws=dict(cumulative=True),

#             kde_kws=dict(cumulative=True),ax=axs[1])



# summary statistics

print(t5['logRet'].describe())
# plot a CDF

#counts, bin_edges = np.histogram(np.array(t5['logRet']), bins=20, normed=True)

#cdf = np.cumsum(counts)

#pylab.plot(bin_edges[1:], cdf)

sns.distplot(t5['logRet'],

             hist_kws=dict(cumulative=True),

             kde_kws=dict(cumulative=True))
#Pairwise plot of variables

sns.pairplot(t5);
g = sns.PairGrid(t5)

g.map_diag(sns.kdeplot)

g.map_offdiag(sns.kdeplot, cmap="Blues_d", n_levels=6);
def tsplot(y, lags=None, figsize=(10, 8), style='bmh'):

    if not isinstance(y, pd.Series):

        y = pd.Series(y)

    with plt.style.context(style):    

        fig = plt.figure(figsize=figsize)

        #mpl.rcParams['font.family'] = 'Ubuntu Mono'

        layout = (3, 2)

        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)

        acf_ax = plt.subplot2grid(layout, (1, 0))

        pacf_ax = plt.subplot2grid(layout, (1, 1))

        qq_ax = plt.subplot2grid(layout, (2, 0))

        pp_ax = plt.subplot2grid(layout, (2, 1))

        y.plot(ax=ts_ax)

        ts_ax.set_title('Time Series Analysis Plots')

        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)

        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)

        sm.qqplot(y, line='s', ax=qq_ax)

        qq_ax.set_title('QQ Plot')        

        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

        plt.tight_layout()

    return 

tsplot(t5['logRet'])
def _get_best_model(TS):

    best_aic = np.inf 

    best_order = None

    best_mdl = None

    pq_rng = range(4) # [0,1,2,3,4]

    for i in pq_rng:

        for j in pq_rng:

            try:

                tmp_mdl = smt.ARIMA(TS, order=(i,0,j)).fit(method='mle', trend='nc')

                tmp_aic = tmp_mdl.aic

                if tmp_aic < best_aic:

                    best_aic = tmp_aic

                    best_order = (i, 0, j)

                    best_mdl = tmp_mdl

            except: continue

    print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))

    #p(sms.diagnostic.acorr_ljungbox(best_mdl.res_id, lags=[20], boxpierce=False))

    return best_aic, best_order, best_mdl



best_aic, best_order, best_mdl = _get_best_model(t5['logRet'])

print(sms.diagnostic.acorr_ljungbox(best_mdl.resid, lags=[20], boxpierce=False))
