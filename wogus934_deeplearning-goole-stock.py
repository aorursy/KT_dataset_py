# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Any results you write to the current directory are saved as output.

import keras

import numpy



x = numpy.array([0, 1, 2, 3, 4])

y = x * 2 + 1



model = keras.models.Sequential()

model.add(keras.layers.Dense(1, input_shape=(1,)))

model.compile('SGD', 'mse')



model.fit(x[:2], y[:2], epochs=1000, verbose=0)



print('Expected:', y[2:])

print('Predicted:', model.predict(x[2:]).flatten())
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.optimizers import Adadelta

from keras.optimizers import Adam

from keras.optimizers import SGD

from keras.layers import Dropout

import numpy as np

#NumPy is the fundamental package for scientific computing with Python

np.random.seed(128)

#Fixing the seed for initializing to random weights

import matplotlib.pyplot as plt

#Matplotlib is a Python 2D plotting library 

import pandas as pd

#Pandas is a Python programming language for data manipulation and analysis.

from sklearn.preprocessing import MinMaxScaler

#Scikit-learn is a machine learning library for the Python programming language.



#Parameter setting

num_train=1258

num_test=21

num_epochs=100

mini_batch=32



#Importing the training set

training_set = pd.read_csv("../input/Google_Stock_Price_Train.csv")

training_set = training_set.iloc[:,1:2].values

#Selecting data by rows and columns by number.

#All rows & 2nd column (column index 1).



# Feature Scaling

sc = MinMaxScaler(feature_range = (0,  1))

#Min-Max scaling (normalization)

training_set = sc.fit_transform(training_set)

#Data is scaled into [0,1] range



# Getting the inputs and the ouputs

X_train = training_set[0:num_train-1]

y_train = training_set[1:num_train]

#Predicted value



#Reshaping

X_train = np.reshape(X_train, (num_train-1, 1, 1))

#Input to the RNN should be a 3D array: (number of inputs, timesteps, input data dim)

#=(X_train.shape[0],X_train.shape[1],1)



#Building the RNN

#RNN Architectures: http://slazebni.cs.illinois.edu/spring17/lec20_rnn.pdf



#Initialising the LSTM

model = Sequential()



#Adding the input layer and the LSTM layer

model.add(LSTM(units = 50, return_sequences = True, input_shape = (1, 1)))

#LSTM: ref. https://medium.com/mlreview/understanding-lstm-and-its-diagrams-37e2f46f1714

#return_sequences = True enables multiple LSTM layers

#input_shape=(timestep, data dim)

#default argument of LSTM activation = tanh

model.add(Dropout(0.1))

model.add(LSTM(units = 50))

model.add(Dropout(0.1))



#Adding the output layer

model.add(Dense(units = 1))



#Compiling the RNN

model.compile(optimizer = 'adam', loss='mean_squared_error')

#Available loss functions: https://keras.io/losses/



# Fitting the RNN to the training set

model.fit(X_train, y_train, batch_size = mini_batch, epochs = num_epochs)



#Making the predictions and visualising the results



# Getting the real stock price

test_set = pd.read_csv("../input/Google_Stock_Price_Test.csv")

real_stock_price = test_set.iloc[:,1:2].values



# Getting the predicted stock price

inputs = real_stock_price

inputs = sc.transform(inputs)

inputs = np.reshape(inputs, (num_test, 1, 1))

predicted_stock_price = model.predict(inputs)

predicted_stock_price = sc.inverse_transform(predicted_stock_price)



# Visualising the results

plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')

plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')

plt.title('Google Stock Price Prediction')

plt.xlabel('Time')

plt.ylabel('Google Stock Price')

plt.legend()

plt.show()