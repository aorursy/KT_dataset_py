import tensorflow as tf

import tensorflow.keras as keras

import numpy as np

import pandas as pd

from sklearn import preprocessing

np.random.seed(10)

import matplotlib.pyplot as plt

import seaborn as sns

import os

import fileinput
df=pd.read_csv('../input/tesla-stock-data-from-2010-to-2020/TSLA.csv')
def augFeatures(df):

  df["Date"] = pd.to_datetime(df["Date"])

  df["year"] = df["Date"].dt.year

  df["month"] = df["Date"].dt.month

  df["date"] = df["Date"].dt.day

  df["day"] = df["Date"].dt.dayofweek

  df = df.drop(["Date"], axis=1)

  return df

  #separate the DateTime to year, month and day
def normalize(df):

  df_norm = df.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))

  return df_norm

  #normalize the data
df_feat=augFeatures(df)

df_norm=normalize(df_feat)
def buildTrain(df, ref_day=5, predict_day=1):

    X_train, Y_train = [], []

    for i in range(df.shape[0]-predict_day-ref_day):

        X_train.append(np.array(df.iloc[i:i+ref_day]))

        Y_train.append(np.array(df.iloc[i+ref_day:i+ref_day+predict_day]["Close"]))

    return np.array(X_train), np.array(Y_train)

    #seperate the data to x_train and y_train

    #the main goal is to predict the  day after the fifth-day price
def splitData(X,Y,rate):

  X_train = X[:int(X.shape[0]*rate)]

  Y_train = Y[:int(Y.shape[0]*rate)]

  X_val = X[int(X.shape[0]*rate):]

  Y_val = Y[int(Y.shape[0]*rate):]

  return X_train, Y_train, X_val, Y_val
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, LSTM
X_train, Y_train = buildTrain(df_norm, 5, 1)

X_train, Y_train, X_val, Y_val = splitData(X_train, Y_train, 0.9)
model = Sequential()

model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))

model.add(Dropout(0.2))

model.add(LSTM(units = 50, return_sequences = True))

model.add(Dropout(0.2))

model.add(LSTM(units = 50))

model.add(Dropout(0.2))

model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

model.summary()

train_history=model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs = 100, batch_size = 128, verbose=2)
import matplotlib.pyplot as plt

def show_train_history(train_history,train,validation):

    plt.plot(train_history.history[train])

    plt.plot(train_history.history[validation])

    plt.title('Train History')

    plt.ylabel(train)

    plt.xlabel('Epoch')

    plt.legend(['train', 'validation'], loc='lower right')

    plt.show()
show_train_history(train_history,'loss','val_loss')
model.evaluate(X_val,Y_val)

predict_y = model.predict(X_val)
def denormalize(train):

  denorm = train.apply(lambda x: x*(np.max(df["Close"])-np.min(df["Close"]))+np.mean(df["Close"]))

  return denorm



Y_val = pd.DataFrame(Y_val)

Y_val = denormalize(Y_val)

predict_y = pd.DataFrame(predict_y)

predict_y = denormalize(predict_y)
import matplotlib.pyplot as plt 

plt.figure(figsize = (18,7))

plt.plot(Y_val, color = 'red', label = 'Real Price')  # red line is real CLOSE price

plt.plot(predict_y, color = 'blue', label = 'Predicted Price')  #blue line is predicted price

plt.title('Price Prediction')

plt.xlabel('Time')

plt.ylabel('Stock Price')

plt.legend()

plt.show()
A = Y_val

B = predict_y

retsA=A-A.shift(1)

retsB=B-B.shift(1)

retsA = np.array(retsA)

retsB = np.array(retsB)

count = 0

for i in range(125):

    if retsA[i+1]>retsA[i] and retsB[i+1]>retsB[i]:

        count = count+1

    elif retsA[i+1]<retsA[i] and retsB[i+1]<retsB[i]:

        count = count+1

    else:

        count = count

accuracy = count / 125
print(accuracy)

print(count)