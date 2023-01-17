# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()



%matplotlib inline



df = pd.read_csv("../input/cryptocurrencypricehistory/bitcoin_price.csv", parse_dates=['Date'])

df.tail()
import matplotlib.dates as mdates

df['Date_mpl'] = df['Date'].apply(lambda x: mdates.date2num(x))



fig, ax = plt.subplots(figsize=(12,8))

sns.tsplot(df.Close.values, time=df.Date_mpl.values, alpha=0.8, color=color[3], ax=ax)

ax.xaxis.set_major_locator(mdates.AutoDateLocator())

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y.%m.%d'))

fig.autofmt_xdate()

plt.xlabel('Date', fontsize=12)

plt.ylabel('Price in USD', fontsize=12)

plt.title("Closing price distribution of bitcoin", fontsize=15)

plt.show()
import matplotlib.ticker as mticker

from matplotlib.finance import candlestick_ohlc



fig = plt.figure(figsize=(12,8))

ax1 = plt.subplot2grid((1,1), (0,0))



temp_df = df[df['Date']>'2017-05-01']

ohlc = []

for ind, row in temp_df.iterrows():

    ol = [row['Date_mpl'],row['Open'], row['High'], row['Low'], row['Close'], row['Volume']]

    ohlc.append(ol)

    

candlestick_ohlc(ax1, ohlc, width=0.4, colorup='#77d879', colordown='#db3f3f')

ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))



plt.xlabel("Date", fontsize=12)

plt.ylabel("Price in USD", fontsize=12)

plt.title("Candlestick chart for Bitcoin", fontsize=15)

plt.subplots_adjust(left=0.09, bottom=0.20, right=0.94, top=0.90, wspace=0.2, hspace=0)

plt.show()


dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')

df1 = pd.read_csv("../input/all-crypto-currencies/crypto-markets.csv",parse_dates=['date'])

#df1 = pd.read_csv("../input/all-crypto-currencies/crypto-markets.csv",parse_dates=['date'], index_col='date',date_parser=dateparse)

df1.head(5)

volume = df1.iloc[:,5]

market = df1.iloc[:6]

volume.head(10)
df1_volume=df1.groupby('date',as_index=False).sum()

df1_volume.shape
df1_volume['volume'].shape


import matplotlib.dates as mdates

df1_volume['date_mpl'] = df1_volume['date'].apply(lambda x: mdates.date2num(x))



fig, ax = plt.subplots(figsize=(12,8))

sns.tsplot(df1_volume.volume.values, time=df1_volume.date_mpl.values, alpha=0.8, color=color[3], ax=ax)

ax.xaxis.set_major_locator(mdates.AutoDateLocator())

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y.%m.%d'))

fig.autofmt_xdate()

plt.xlabel('Date', fontsize=12)

plt.ylabel('Market Size', fontsize=12)

plt.title("total market size of five years", fontsize=15)

plt.show()
import matplotlib.dates as mdates

df1_volume['date_mpl'] = df1_volume['date'].apply(lambda x: mdates.date2num(x))



fig, ax = plt.subplots(figsize=(12,8))

sns.tsplot(df1_volume.market.values, time=df1_volume.date_mpl.values, alpha=0.8, color='blue', ax=ax)

ax.xaxis.set_major_locator(mdates.AutoDateLocator())

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y.%m.%d'))

fig.autofmt_xdate()

plt.xlabel('Date', fontsize=12)

plt.ylabel('Total Market Value', fontsize=12)

plt.title("Total Market Value for five years", fontsize=15)

plt.show()
df2 = pd.read_excel("../input/question3/question3.xlsx")

BlockChainCompanies=df2.iloc[:,0]

Scores = df2.iloc[:,1]
width = 1.0

fig, ax = plt.subplots(figsize=(10,18))

ind = np.arange(len(Scores))

rects = ax.barh (ind,Scores,height =0.5, color='y',alpha = 0.8)

plt.gca().invert_yaxis()

ax.set_yticks(ind)

ax.set_yticklabels(BlockChainCompanies, rotation='horizontal',)

ax.set_xlabel("Scores")

ax.set_title("BlockChainCompanies")

plt.show()

fig = figsize=(15,20)

plt.figure()

x = df2.iloc[0:10,0]

y = df2.iloc[0:10,1]

df2.iloc[0:10].plot(x= df2.iloc[0:10,0],kind='barh',title="Top 10 Blockchain Compainese",colormap='Paired')

#prediction

eth = pd.read_csv("../input/cryptocurrencypricehistory/ethereum_price.csv", parse_dates=['Date'])

eth.tail()

ltc = pd.read_csv("../input/cryptocurrencypricehistory/litecoin_price.csv",parse_dates=['Date'])

iota = pd.read_csv("../input/cryptocurrencypricehistory/iota_price.csv",parse_dates=['Date'])

monero=pd.read_csv("../input/cryptocurrencypricehistory/monero_price.csv",parse_dates=['Date'])

nem  = pd.read_csv("../input/cryptocurrencypricehistory/nem_price.csv",parse_dates=['Date'])

numeraire = pd.read_csv("../input/cryptocurrencypricehistory/numeraire_price.csv",parse_dates=['Date'])

omisego = pd.read_csv("../input/cryptocurrencypricehistory/omisego_price.csv",parse_dates=['Date'])

qtum = pd.read_csv("../input/cryptocurrencypricehistory/qtum_price.csv",parse_dates=['Date'])

riipple = pd.read_csv("../input/cryptocurrencypricehistory/ripple_price.csv",parse_dates=['Date'])

stratis = pd.read_csv("../input/cryptocurrencypricehistory/stratis_price.csv",parse_dates=['Date'])

waves= pd.read_csv("../input/cryptocurrencypricehistory/waves_price.csv",parse_dates=['Date'])
iota = iota[['Date','Close']]

monero = monero[['Date','Close']]

nem = nem[['Date','Close']]

numeraire = numeraire[['Date','Close']]

omisego = omisego[['Date','Close']]

qtum = qtum[['Date','Close']]

riipple = riipple[['Date','Close']]

stratis = stratis[['Date','Close']]

waves = waves[['Date','Close']]
other = pd.merge(iota,monero,left_index=True,right_index=True,how='inner')
