import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import tensorflow as tf
import keras
ds = pd.read_csv('/kaggle/input/foreign-exchange-rates-per-dollar-20002019/Foreign_Exchange_Rates.csv', na_values='ND')
ds
ds.isnull().sum()
ds.interpolate(inplace=True)
ds.isnull().sum()
plt.plot(ds['UNITED KINGDOM - UNITED KINGDOM POUND/US$'])
plt.plot(ds['CANADA - CANADIAN DOLLAR/US$'])
df = ds['CANADA - CANADIAN DOLLAR/US$']
df
df = np.array(df).reshape(-1,1)
df
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df = scaler.fit_transform(df)
df
train = df[:4800]
test = df[4800:]

train.shape , test.shape
def get_data(data, look_back):
  datax, datay = [],[]
  for i in range(len(data)-look_back-1):
    datax.append(data[i:(i+look_back),0])
    datay.append(data[i+look_back,0])
  return np.array(datax) , np.array(datay)
look_back = 1

x_train , y_train = get_data(train, look_back)
x_train.shape , y_train.shape
x_test , y_test = get_data(test,look_back)
x_test.shape, y_test.shape
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1], 1)

x_train.shape, x_test.shape
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(5, activation='tanh', input_dim = 1))
model.add(tf.keras.layers.Dense(1))
model.compile(optimizer='adam', loss = 'mse')
model.fit(x_train,y_train, epochs = 5, batch_size=1)
scaler.scale_
y_pred = model.predict(x_test)

y_pred = scaler.inverse_transform(y_pred)
y_pred[:10]
y_test = np.array(y_test).reshape(-1,1)
y_test = scaler.inverse_transform(y_test)
y_test[:10]
plt.plot(y_test , label = 'Actual')
plt.plot(y_pred , label = 'Predicted')
plt.legend()