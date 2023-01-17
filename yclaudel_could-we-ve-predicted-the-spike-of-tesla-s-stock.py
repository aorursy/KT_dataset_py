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
# Calculate the daily return

daily_rtn_all = df.loc[df.index>'2018-12-31','Close']/df.loc[df.index>'2018-12-31','Close'].shift(1) - 1

daily_rtn_all.dropna(inplace=True)

daily_rtn = daily_rtn_all[daily_rtn_all.index<'2019-12-19']

# plot the distribution

import seaborn as sns

sns.distplot(daily_rtn)

plt.show()

print( 'Skewness (before)  =', daily_rtn.skew())

print( 'Skewness (overall) =', daily_rtn_all.skew())
print( 'Kurtosis (before)  =', daily_rtn.kurt())

print( 'Kurtosis (overall) =', daily_rtn_all.kurt())
print( 'Before 2019-12-19')

print( 'Historical VaR at 1 days at 95%  =', daily_rtn.sort_values().quantile(0.05))

print( 'Historical Mirror VaR at 1 days at 5%  =', daily_rtn.sort_values().quantile(0.95))

print( 'Overall')

print( 'Historical VaR at 1 days at 95%  =', daily_rtn_all.sort_values().quantile(0.05))

print( 'Historical Mirror VaR at 1 days at 5%  =', daily_rtn_all.sort_values().quantile(0.95))
horizon = 20

nbr_simulation = 10000



std = np.std(daily_rtn)

mean = np.mean(daily_rtn)



gen=1 + mean + std * np.random.randn(nbr_simulation,horizon) 

gen = np.prod(gen,axis=1)-1

gen.sort()
print('Mean of daily return = {}'.format(mean))

print('Standard deviation of daily return = {}'.format(std))

print( 'VaR at {} days at 95%  = {}'.format(horizon,np.quantile(gen, 0.05)))

print( 'Mirror VaR at {} days at 95%  = {}'.format(horizon,np.quantile(gen, 0.95)))
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.layers import Bidirectional, Dropout, Activation, Dense, LSTM

from tensorflow.python.keras.layers import CuDNNLSTM

from tensorflow.keras.models import Sequential

import tensorflow as tf

from tensorflow import keras

import math



# Window size = number of previous values to predict the next value

WINDOW_SIZE = 10
close_price = df.Close[(df.index>'2017-12-31')]

close_price['2019-12-19']

train_close = df.Close[(df.index>'2017-12-31') & (df.index<'2019-12-23')].values.reshape(-1, 1)

all_close = df.Close[df.index>'2017-12-31'].values.reshape(-1, 1)

#  MinMaxScaler

scaler = MinMaxScaler()

scaled_all_close = scaler.fit_transform(all_close)

scaled_train_close = scaler.transform(train_close)                     

#scaled_close = scaled_close[~np.isnan(scaled_close)]

#scaled_close = scaled_close.reshape(-1, 1)

print("Train shape = {}".format(scaled_train_close.shape))

print("All shape = {}".format(scaled_all_close.shape))

print("nan values ? {}".format(np.isnan(scaled_train_close).any()))
# Generate sequences of lenght = WINDOW_SIZE



def generateSequence(sequence,backward):

    X, y = list(), list()

    for i in range(sequence.shape[0]-backward):

        seq_x, seq_y = sequence[i:i+backward], sequence[i+backward]

        X.append(seq_x)

        y.append(seq_y)

    X=np.array(X)

    y=np.array(y)

    X = X.reshape((X.shape[0], X.shape[1], 1))

    return X,y

    

X,y = generateSequence(scaled_train_close,WINDOW_SIZE)

print("X shape = {}".format(X.shape))

print("y shape = {}".format(y.shape))
model = Sequential()

model.add(LSTM(12, activation='relu', input_shape=(WINDOW_SIZE, 1)))

model.add(Dense(8, activation='relu'))

model.add(Dense(1))
#  Compile

model.compile(

    loss='mse', 

    optimizer='adam'

)



BATCH_SIZE = 64



#  Compile

history = model.fit(

    X, 

    y, 

    epochs=50, 

    batch_size=BATCH_SIZE, 

    shuffle=False,

    validation_split=0.1

)

trainScore = model.evaluate(X, y)

print('Train Score: %.6f MSE (%.6f RMSE)' % (trainScore, math.sqrt(trainScore)))



plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
y_train_predicted = model.predict(X)

y_inverse = scaler.inverse_transform(y)

y_train_predicted_inverse = scaler.inverse_transform(y_train_predicted)





plt.plot(y_inverse.ravel(), label="Price", color='black')

plt.plot(y_train_predicted_inverse.ravel(), label="Predicted Price", color='blue')

plt.legend(loc='upper left')

plt.title("Train data - Prediction at 1 day based on the previous {} days".format(WINDOW_SIZE))

plt.show()
X,y = generateSequence(scaled_all_close,WINDOW_SIZE)



y_predicted = model.predict(X)

y_inverse = scaler.inverse_transform(y)

y_predicted_inverse = scaler.inverse_transform(y_predicted)





plt.plot(y_inverse.ravel(), label="Price", color='black')

plt.plot(pd.Series(y_predicted_inverse[:487].ravel(),index=range(0,487)), label="Train Predicted Price", color='blue')

plt.plot(pd.Series(y_predicted_inverse[487:].ravel(),index=range(487,515)), label="Test Predicted Price", color='red')



plt.legend(loc='upper left')

plt.title("All data - Prediction at 1 day based on the previous {} days".format(WINDOW_SIZE))

plt.show()