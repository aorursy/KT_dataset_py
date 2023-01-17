# Import librairies

%matplotlib inline 

import matplotlib.pylab

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

plt.style.use('seaborn')

plt.rcParams['figure.figsize'] = [20, 9]

plt.rcParams['lines.linewidth'] = 1
# Load the data

df = pd.read_csv("../input/tesla-stock-data-from-2010-to-2020/TSLA.csv")

df['Date'] = pd.to_datetime(df['Date'])

df.index = df['Date']

df.tail(10)
def plot_with_rolling(data,short_rolling,long_rolling,lbl='Close Price'):

    # Plot 

    fig, ax = plt.subplots(figsize=(20,9))



    ax.plot(data.index, data, label=lbl)

    ax.plot(short_rolling.index, short_rolling, label='20 days rolling')

    ax.plot(long_rolling.index, long_rolling, label='100 days rolling')

    ax.axvline(x='2019-12-19',linewidth=0.8, color='r')

    ax.axvline(x='2019-06-01',linewidth=0.8, color='g')

    ax.set_xlabel('Date')

    ax.set_ylabel(lbl)

    ax.legend()



    plt.show()



short = df.Close.rolling(window=20).mean()

long = df.Close.rolling(window=100).mean() 

plot_with_rolling(data=df.Close,short_rolling=short,long_rolling=long,lbl='Close Price')    

    

short = df.Close[df.index>'2018-12-21'].rolling(window=20).mean()

long = df.Close[df.index>'2018-12-21'].rolling(window=100).mean() 

plot_with_rolling(data=df.Close[df.index>'2018-12-21'],short_rolling=short,long_rolling=long,lbl='Close Price')
short = df.Volume.rolling(window=20).mean()

long = df.Volume.rolling(window=100).mean()

plot_with_rolling(data=df.Volume,short_rolling=short,long_rolling=long,lbl='Volume')    

    

short = df.Volume[df.index>'2018-12-21'].rolling(window=20).mean()

long = df.Volume[df.index>'2018-12-21'].rolling(window=100).mean() 

plot_with_rolling(data=df.Volume[df.index>'2018-12-21'],short_rolling=short,long_rolling=long,lbl='Volume')

# Calculate the daily return and the daily log return

daily_rtn = df.Close/df.Close.shift(1) - 1

daily_log_rtn = np.log(df.Close).diff()

# Calculate the daily return (another way)

# daily_rtn = close.pct_change(1)



daily_rtn.plot()

plt.show()

# log return

daily_rtn.rolling(30).var().plot()

plt.show()
df['Volume'].rolling(50).var().plot()

plt.show()
# Calculate diff

diff=df.Close.diff().dropna()

# Ups

plus=diff.map(lambda x: x if x>0 else 0).rename('Plus')

# Downs

minus=diff.map(lambda x: -1*x if x<0 else 0).rename('Minus')



rsi14=pd.concat([diff,plus,minus],axis=1)

# init average up

rsi14['AvgUP'] = rsi14.Plus[:14].sum()/14

# init average down

rsi14['AvgDOWN'] = rsi14.Minus[:14].sum()/14



# calculate AvgUP and AvgDown with the recurrent formula

for i in range(14,rsi14.Close.size):

    rsi14['AvgUP'].iloc[i]=(rsi14['Plus'].iloc[i] + 13*rsi14['AvgUP'].iloc[i-1])/14

    rsi14['AvgDOWN'].iloc[i]=(rsi14['Minus'].iloc[i] + 13*rsi14['AvgDOWN'].iloc[i-1])/14



rsi14['RSI']=100 - 100 / (1+rsi14['AvgUP']/rsi14['AvgDOWN'])

rsi14.tail(10)
short = rsi14['RSI'].rolling(window=20).mean()

long = rsi14['RSI'].rolling(window=100).mean()

plot_with_rolling(data=rsi14['RSI'],short_rolling=short,long_rolling=long,lbl='Close Price')    
ema_short = df.Close.ewm(span=12, adjust=False).mean()

ema_long = df.Close.ewm(span=26, adjust=False).mean()

macd_line = ema_short - ema_long

macd_signal = macd_line.ewm(span=9, adjust=False).mean()

# Plot 

fig, ax = plt.subplots(figsize=(20,9))



ax.plot(macd_line.index, macd_line, label='MACD Line')

ax.plot(macd_signal.index, macd_signal, label='MACD Signal')

ax.axvline(x='2019-12-19',linewidth=0.8, color='r')

ax.set_xlabel('Date')

ax.set_ylabel('MACD')

ax.legend()



plt.show()

momentum = df.Close.diff(20)

momentum.plot()

plt.show()


def plot_all(close=df.Close,rsi=rsi14['RSI'],macd=macd_line,momentum=momentum):

    fig = plt.figure(constrained_layout=True,figsize=(20,16))

    gs = fig.add_gridspec(10, 1)

    ax1 = fig.add_subplot(gs[:4, 0])

    ax1.set_title('Close')

    ax1.plot(close.index, close, label='Close')

    ax1.axvline(x='2019-12-19',linewidth=0.8, color='r')

    ax2 = fig.add_subplot(gs[4:6,0])

    ax2.set_title('RSI14')

    ax2.plot(rsi.index, rsi)

    ax2.axhline(y=30,linewidth=0.5, color='blue')

    ax2.axhline(y=70,linewidth=0.5, color='red')

    ax2.axvline(x='2019-12-19',linewidth=0.8, color='r')

    ax3 = fig.add_subplot(gs[6:8,0])

    ax3.set_title('MACD')    

    ax3.plot(macd.index, macd)

    ax3.axhline(y=0,linewidth=0.5, color='black')

    ax3.axvline(x='2019-12-19',linewidth=0.8, color='r')

    ax4 = fig.add_subplot(gs[8:,0])

    ax4.set_title('Momentum')

    ax4.plot(momentum.index, momentum)

    ax4.axhline(y=0,linewidth=0.5, color='black')

    ax4.axvline(x='2019-12-19',linewidth=0.8, color='r')

    plt.show()

plot_all()

startd = '2018-12-31'

plot_all(close=df.Close[df.index>startd],rsi=rsi14['RSI'][rsi14.index>startd],macd=macd_line[macd_line.index>startd],momentum=momentum[momentum.index>startd])