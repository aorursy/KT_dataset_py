# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()



%matplotlib inline



pd.options.mode.chained_assignment = None

pd.options.display.max_columns = 999



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import sys 

print(sys.version)



bitcoin = pd.read_csv('../input/bitcoin_price.csv', parse_dates= ['Date'])

ethereum = pd.read_csv('../input/ethereum_price.csv', parse_dates= ['Date'])

ethereum_classic = pd.read_csv('../input/ethereum_classic_price.csv', parse_dates= ['Date'])
bitcoin.columns
bitcoin[['Date','Close','Volume','Market Cap']].head()
ethereum[['Date','Close','Volume','Market Cap']].head()
ethereum_classic[['Date','Close','Volume','Market Cap']].head()
mosthigh = np.array(bitcoin[['High']][(bitcoin['Date'] >= 'jan 01, 2016') & (bitcoin['Date'] <= 'dec 31, 2016')])
mostlow = np.array(bitcoin[['Low']][(bitcoin['Date'] >= 'jan 01, 2016') & (bitcoin['Date'] <= 'dec 31, 2016')])
dateoftime = np.array(bitcoin[['Date']][(bitcoin['Date'] >= 'jan 01, 2016') & (bitcoin['Date'] <= 'dec 31, 2016')])
x = mosthigh

z = mostlow

y = dateoftime

difference = x - z

u = difference
np.sort(u, axis=None)
table = []

for u, y in zip(difference, dateoftime):

    table.append([u, y])

table
sorted(table1)
print('*** Bitcoin ***' , '\n', bitcoin[['Date', 'Close']].sort_values('Close', ascending = False).head(1), '\n')

print('*** Ethereum ***' , '\n',ethereum[['Date', 'Close']].sort_values('Close', ascending = False).head(1), '\n')

print('*** Ethereum_classic ****', '\n', ethereum_classic[['Date', 'Close']].sort_values('Close', ascending = False).head(1))
print('*** Bitcoin ***' , '\n', bitcoin[['Date', 'Close']].sort_values('Close', ascending = False).tail(1), '\n')

print('*** Ethereum ***' , '\n',ethereum[['Date', 'Close']].sort_values('Close', ascending = False).tail(1), '\n')

print('*** Ethereum_classic ****', '\n', ethereum_classic[['Date', 'Close']].sort_values('Close', ascending = False).tail(1))
print('*** Bitcoin ***' , '\n', bitcoin[['Date', 'Close']].tail(1), '\n')

print('*** Ethereum ***' , '\n',ethereum[['Date', 'Close']].tail(1), '\n')

print('*** Ethereum_classic ****', '\n', ethereum_classic[['Date', 'Close']].tail(1))
import matplotlib.dates as mdates

bitcoin['Date_mpl'] = bitcoin['Date'].apply(lambda x: mdates.date2num(x))

fig, ax = plt.subplots(figsize=(12,6))

sns.tsplot(bitcoin.Open.values, time=bitcoin.Date_mpl.values, alpha=0.8, color=color[3], ax=ax)

ax.xaxis.set_major_locator(mdates.AutoDateLocator())

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y.%m.%d'))

fig.autofmt_xdate()

plt.xlabel('Date', fontsize=12)

plt.ylabel('Price in USD', fontsize=12)

plt.title("Closing price distribution of Bitcoin", fontsize=15)

plt.show()
themarketcap = np.array(bitcoin[['Market Cap']][(bitcoin['Date'] >= 'may 02, 2013') & (bitcoin['Date'] <= '3 oct, 2017')])
dateoftimemc = np.array(bitcoin[['Date']][(bitcoin['Date'] >= 'may 02, 2013') & (bitcoin['Date'] <= '3 oct, 2017')])
d = themarketcap

e = dateoftimemc
np.sort(d, axis=None)
table2 = []

for d, e in zip(themarketcap, dateoftimemc):

    table2.append([d, e])

table2
sorted(table2)
min(table2)
import matplotlib.ticker as mticker

from matplotlib.finance import candlestick_ohlc



fig = plt.figure(figsize=(12,8))

ax1 = plt.subplot2grid((1,1), (0,0))



temp_df = bitcoin[(bitcoin['Market Cap'] > '01-06-2017')]

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
crypto['bitcoin']=pd.read_csv('../input/bitcoin_price.csv')



df = pd.read_csv('../input/bitcoin_price.csv', parse_dates=['Date'])

import matplotlib.dates as mdates

df['Date_mpl'] = df['Date'].apply(lambda x: mdates.date2num(x))



fig, ax = plt.subplots(figsize=(12,8))

sns.tsplot(df.Market_Cap.values, time=df.Date_mpl.values, alpha=0.8, color=color[3], ax=ax)

ax.xaxis.set_major_locator(mdates.AutoDateLocator())

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y.%m.%d'))

fig.autofmt_xdate()

plt.xlabel('Date', fontsize=12)

plt.ylabel('Price in USD', fontsize=12)

plt.title("Closing price distribution of Ethereum", fontsize=15)

plt.show()
import matplotlib.ticker as mticker

from matplotlib.finance import candlestick_ohlc



stock_data = []

split_source = bitcoin

for line in split_source[1:]:

    split_line = line.split(',')

    if len(split_line) == 7:

        if 'values' not in line and 'labels' not in line:

            stock_data.append(line)

            

import matplotlib.dates as mdates

ethereum['Date_mpl'] = ethereum['Date'].apply(lambda x: mdates.date2num(x))

fig, ax = plt.subplots(figsize=(12,8))

sns.tsplot(ethereum.Close.values, time=ethereum.Date_mpl.values, alpha=0.8, color=color[3], ax=ax)

ax.xaxis.set_major_locator(mdates.AutoDateLocator())

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y.%m.%d'))

fig.autofmt_xdate()

plt.xlabel('Date', fontsize=12)

plt.ylabel('Price in USD', fontsize=12)

plt.title("Closing price distribution of Ethereum", fontsize=15)

plt.show()
import matplotlib.ticker as mticker

from matplotlib.finance import candlestick_ohlc



fig = plt.figure(figsize=(12,8))

ax1 = plt.subplot2grid((1,1), (0,0))



temp_df = bitcoin[(bitcoin['Market Cap']> '2013-05-01' )] 

ohlc = []

for ind, row in temp_df.iterrows():

    ol = row['Market Cap']

    ohlc.append(ol)

    

candlestick_ohlc(ax1, ohlc, width=0.4, colorup='#77d879', colordown='#db3f3f')

ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))



plt.xlabel("Market Cap", fontsize=12)

plt.ylabel("Price in USD", fontsize=12)

plt.title("Candlestick chart for Bitcoin from 01-01-2017", fontsize=15)

plt.subplots_adjust(left=0.09, bottom=0.20, right=0.94, top=0.90, wspace=0.2, hspace=0)

plt.show()
import matplotlib.dates as mdates

ethereum_classic['Date_mpl'] = ethereum_classic['Date'].apply(lambda x: mdates.date2num(x))

fig, ax = plt.subplots(figsize=(12,8))

sns.tsplot(ethereum_classic.Close.values, time=ethereum_classic.Date_mpl.values, alpha=0.8, color=color[3], ax=ax)

ax.xaxis.set_major_locator(mdates.AutoDateLocator())

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y.%m.%d'))

fig.autofmt_xdate()

plt.xlabel('Date', fontsize=12)

plt.ylabel('Price in USD', fontsize=12)

plt.title("Closing price distribution of Ethereum Classic", fontsize=15)

plt.show()
percent_change = []

change = []

Sevendays_change = []

price_7days_before = bitcoin['Open'][0]

for ind,row in bitcoin.iterrows():

    if ind > 7:

        price_7days_before = bitcoin['Open'][ind-7]

    change.append(row['Close'] - row['Open'])

    percent_change.append((row['Close'] - row['Open'])/row['Open'])

    Sevendays_change.append((row['Close'] - price_7days_before)/price_7days_before)

bitcoin['Change'] = change

bitcoin['percent_change'] = percent_change

bitcoin['Sevendays_change'] = Sevendays_change

bitcoin.head()
#change graph

import matplotlib.dates as mdates

import seaborn as sns

import matplotlib.pyplot as plt

color = sns.color_palette()

bitcoin['Date_mpl'] = bitcoin['Date'].apply(lambda x: mdates.date2num(x))

fig, ax = plt.subplots(figsize=(12,8))

sns.tsplot(bitcoin.percent_change.values, time=bitcoin.Date_mpl.values, alpha=0.8, color=color[3], ax=ax)

ax.xaxis.set_major_locator(mdates.AutoDateLocator())

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y.%m.%d'))

fig.autofmt_xdate()

plt.xlabel('Date', fontsize=12)

plt.ylabel('Percent change', fontsize=12)

plt.title("Change distribution of bitcoin", fontsize=15)

plt.show()
files_to_use = ["bitcoin_price.csv", "ethereum_price.csv", "neo_price.csv","stratis_price.csv","dash_price.csv"]



cols_to_use = []

for ind, file_name in enumerate(files_to_use):

    currency_name = file_name.split("_")[0]

    if ind == 0:

        df = pd.read_csv("../input/"+file_name, usecols=["Date", "Close"], parse_dates=["Date"])

        df.columns = ["Date", currency_name]

    else:

        temp_df = pd.read_csv("../input/"+file_name, usecols=["Date", "Close"], parse_dates=["Date"])

        temp_df.columns = ["Date", currency_name]

        df = pd.merge(df, temp_df, on="Date")

    cols_to_use.append(currency_name)

df.head()

        

temp_df = df[cols_to_use]

corrmat = temp_df.corr(method='spearman')

fig, ax = plt.subplots(figsize=(10, 10))

sns.heatmap(corrmat, vmax=1., square=True)

plt.title("Cryptocurrency correlation map", fontsize=15)

plt.show()