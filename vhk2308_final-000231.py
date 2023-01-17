# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
TRAIN_CSV = "/kaggle/input/cell-000231/train_cell_000231.xlsx"



dataset = pd.read_excel(TRAIN_CSV)

dataset.info

dataset["Date"] = pd.to_datetime(dataset.Date.astype(str))

dataset["Hour"] = pd.to_timedelta(dataset.Hour, unit="h")

dataset["DateTime"] = pd.to_datetime(dataset.Date + dataset.Hour)

dataset = dataset.drop(["Hour", "Date"], axis=1)

dataset.head
dataset = dataset.set_index("DateTime")

dataset = dataset.sort_index()

dataset.head()
dataset = dataset.drop(["CellName"], axis=1)

dataset.head
train,test = dataset[:8400],dataset[8400:-2]

train,test = train["Traffic"].values,test["Traffic"].values
train =train.reshape(-1, 1)

test =test.reshape(-1, 1)

train.shape
len(train),len(test)
import numpy

import matplotlib.pyplot as plt

import pandas

import math

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error
scaler = MinMaxScaler(feature_range=(0, 1))

train_scaled = scaler.fit_transform(train)
# Creating a data structure with 24 timesteps and 1 output

X_train = []

y_train = []

for i in range(60, len(train)):

    X_train.append(train_scaled[i-60:i, 0])

    y_train.append(train_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
#reshape

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

y_train
len(test)
from keras.layers import Dropout



regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))

regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))

regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))

regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))

regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor.fit(X_train, y_train, epochs = 25, batch_size = 42)
inputs = dataset["Traffic"]

inputs = inputs[len(inputs) - len(test) - 60:].values

inputs = inputs.reshape(-1, 1)

len(inputs)
inputs = scaler.transform(inputs)

len(inputs)
X_test = []

y_test = []

for i in range(60, len(inputs)):

    X_test.append(inputs[i-60:i,:])

    y_test.append(inputs[i])

X_test, y_test = np.array(X_test), np.array(y_test)

y_test
y_test = y_test.reshape(-1,1)

y_test = scaler.inverse_transform(y_test)
X_test.shape
predicts = regressor.predict(X_test)

predicts = scaler.inverse_transform(predicts)

predicts
plt.plot(y_test, color = 'red', label = 'Real LTE Traffic ')

plt.plot(predicts, color = 'blue', label = 'Predicted LTE Traffic')

plt.title('LSTM Based LTE Traffic Predictor')

plt.xlabel('Input from Test CSV')

plt.ylabel('Traffic')

plt.legend()

plt.show()
#sliding window creation

#for predicting next value we need before 60 values as we have taken a 60 timestamps.

#As an input for predection we will pass a sample of time stamp and how many predictions as well.

#define input as n
input_sample = X_test[-1].reshape(-1,1)

#input_sample = np.reshape(input_sample, (1, input_sample.shape[0], 1))

input_sample
input_sample
"""input_sample = input_sample[:,1:,:]

input_sample

test = np.array([[[66]]])

input_sample = np.concatenate((input_sample,test),axis=1)

input_sample.shape"""
def sliding_window(model, scalar, n_predict, input_sample):

    #input_sample = scaler.transform(input_sample)

    input_sample = np.reshape(input_sample, (1, input_sample.shape[0], 1))

    list_pred = []

    for i in range(n_predict):

        predict = model.predict(input_sample)

        print(predict)

        a = scaler.inverse_transform(predict)

        print(a)

        list_pred.append(a[0][0].tolist())

        #predict = scaler.transform(predict)

        predict = np.reshape(predict, (1, predict.shape[0], 1))

        input_sample = input_sample[:,1:,:]

        input_sample = np.concatenate((input_sample,predict),axis=1)

        print(input_sample)

        

    return list_pred

        
a = sliding_window(regressor, scaler, 7, input_sample)
a
x = [1, 2, 3, 4, 5, 6, 7]

plt.plot(x, a, color = 'blue', label = 'Predicted LTE Traffic')

plt.title("Prediction for next n mins")

plt.xlabel('Time')

plt.ylabel('Traffic')

plt.legend()

plt.show()