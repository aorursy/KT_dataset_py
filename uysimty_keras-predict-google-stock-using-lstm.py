import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt 



import warnings

warnings.filterwarnings('ignore')



import os

print(os.listdir("../input"))
stock_data = pd.read_csv("../input/Google_Stock_Price_Train.csv")

test_data = pd.read_csv("../input/Google_Stock_Price_Test.csv")
stock_data.head()
stock_data.tail()
stock_data.info()
stock_data['Date'] = pd.to_datetime(stock_data['Date'])

stock_data = stock_data.sort_values(by=['Date'], ascending=True).reset_index()
stock_data.head()
stock_data.tail()
plt.figure(figsize=(18, 8))

plt.plot(stock_data['Open'])

plt.title("Google Stock Prices")

plt.xlabel("Time (oldest -> latest)")

plt.ylabel("Stock Opening Price")

plt.show()
plt.figure(figsize=(18, 8))

plt.plot(stock_data['High'])

plt.title("Google Stock Prices")

plt.xlabel("Time (oldest-> latest)")

plt.ylabel("Stock Hightest Points")

plt.show()
plt.figure(figsize=(18, 8))

plt.plot(stock_data['Low'])

plt.title("Google Stock Prices")

plt.xlabel("Time (oldest -> latest)")

plt.ylabel("Stock Lowest Points")

plt.show()
plt.figure(figsize=(18, 8))

plt.plot(stock_data['Volume'])

plt.title("Volume of stocks sold")

plt.xlabel("Time (oldest-> latest)")

plt.ylabel("Volume of stocks traded")

plt.show()
input_feature = stock_data[['Open', 'High', 'Low', 'Volume', 'Close']]

input_data = input_feature.values
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))

input_data[:,:] = scaler.fit_transform(input_data[:,:])
lookback=50

total_size=len(stock_data)

X=[]

y=[]

for i in range(0, total_size-lookback): # loop data set with margin 50 as we use 50 days data for prediction

    t=[]

    for j in range(0, lookback): # loop for 50 days

        current_index = i+j

        t.append(input_data[current_index, :]) # get data margin from 50 days with marging i

    X.append(t)

    y.append(input_data[lookback+i, 4])
test_size=100 # 100 days for testing data

X, y= np.array(X), np.array(y)

X_test = X[:test_size]

Y_test = y[:test_size]



X_work = X[test_size:]

y_work = y[test_size:]



validate_size = 10



X_valid = X[:validate_size]

y_valid = y[:validate_size]

X_train = X[validate_size:]

y_train = y[validate_size:]
X_train = X_train.reshape(X_train.shape[0], lookback, 5)

X_valid = X_valid.reshape(X_valid.shape[0], lookback, 5)

X_test = X_test.reshape(X_test.shape[0], lookback, 5)

print(X_train.shape)

print(X_valid.shape)

print(X_test.shape)
from keras import Sequential

from keras.layers import Dense, LSTM



model = Sequential()

model.add(LSTM(50, return_sequences= True, activation='relu', input_shape=(X.shape[1], 5)))

model.add(LSTM(50, return_sequences=True, activation='relu'))

model.add(LSTM(50))

model.add(Dense(1))

model.summary()
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau



callbacks = [

    EarlyStopping(patience=10, verbose=1),

    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),

    ModelCheckpoint('model.h5', verbose=1, save_best_only=True, save_weights_only=True)

]
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_valid, y_valid), callbacks=callbacks)
predicted_value = model.predict(X_test)
plt.figure(figsize=(18, 8))

plt.plot(predicted_value, color= 'red')

plt.plot(Y_test, color='green')

plt.title("Close price of stocks sold")

plt.xlabel("Time (latest ->oldest-> )")

plt.ylabel("Stock Opening Price")

plt.show()