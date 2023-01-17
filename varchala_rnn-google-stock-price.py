# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 06:56:31 2020

@author: G.Varchaleswari
"""

import numpy as np
import matplotlib.pyplot as pt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import pandas as pd

#%%import the data
stock_data = pd.read_csv('../input/google-stock-price/Google_Stock_Price_Train.csv')
data_train = stock_data.iloc[:,1:2]
data_train = np.array(data_train)


#%%feature scaling
msc = MinMaxScaler()
data_train = msc.fit_transform(data_train)
#%% Creating a data structure with 60 timestaps and 1 output
x_train = []
y_train = []

for i in range(60, 1258):
    x_train.append(data_train[i-60:i, 0])
    y_train.append(data_train[i,0])
    

x_train = np.array(x_train)
y_train = np.array(y_train)    

#%% reshaping
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#%% building the rnn

model = tf.keras.models.Sequential()

#%% Adding LSTM layer and the regularization dropout
model.add(tf.keras.layers.LSTM(units =  70, return_sequences = True, input_shape = (x_train.shape[1],1)))
model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.LSTM(units =  70, return_sequences = True))
model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.LSTM(units =  70, return_sequences = True))
model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.LSTM(units =  70))
model.add(tf.keras.layers.Dropout(0.25))

#%% adding the output layer
model.add(tf.keras.layers.Dense(units = 1))

#%% compile the model
model.compile(optimizer = 'adam', loss='mean_squared_error')


#%% fit the model
model.fit(x_train, y_train, epochs = 100, batch_size = 32 )

#%% prediction
dataset_test = pd.read_csv('../input/google-stock-price/Google_Stock_Price_Test.csv')
dataset_test = dataset_test.iloc[:, 1:2].values

dataset_total = pd.concat([pd.DataFrame(data_train), pd.DataFrame(dataset_test)], axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = msc.transform(inputs)
x_test = []
for i in range(60, 80):
    x_test.append(inputs[i-60:i,0])
    
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
predicted_outcome = model.predict(x_test)
predicted_outcome = msc.inverse_transform(predicted_outcome)

#%% Plot the actual values vs the predicted outcome of stock price on a graph

pt.xlabel('Time')
pt.ylabel('Stock Price')
pt.plot(predicted_outcome, color = 'green', label = 'Predicted Stock Price' )
pt.plot(dataset_test, color = 'blue', label = 'Actual Stock Price' )
pt.legend()
pt.show()
