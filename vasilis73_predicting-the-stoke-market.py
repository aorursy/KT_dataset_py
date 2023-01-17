# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Import SPY data
spy = pd.read_csv("../input/index-sp/spy2.csv")
spy.head()
#spy.shape
#spy.tail()
spy['Adj Close'].tail(60).min()
spy['Adj Close'].tail(60).mean()
spy['Adj Close'].tail(60).max()
spy['Adj Close'].std()
df = pd.read_csv("../input/index-sp/spy2.csv",index_col='Date')
df.head()
import matplotlib.pyplot as plt
%matplotlib inline
df['Adj Close'].plot(figsize=(16,6))
plt.axhline(y=2583, color='g', linestyle='-')
plt.axhline(y=2357, color='r', linestyle='-')
plt.axhline(y=2873, color='r', linestyle='-')
spy.head()
spy.corr()
dataset = spy.drop('Close',axis=1)
dataset.head()
dataset.shape
dataset_train = dataset.iloc[0:200,1:5]
dataset_train.head()
dataset_train.shape
dataset_test = dataset.iloc[200:252,1:5]
dataset_test.head()
dataset_test.shape
training_set = dataset_train.values
# perform feature scaling 
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 200):
    X_train.append(training_set_scaled[i-60:i, :])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 4))
# Building the RNN
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
# Build and fit the RNN
# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 4)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
# Making predictions and visualising the results
# Getting the real SPY prices for the period 28/03/2018 to 11/05/2018
real_spy_price = dataset_test['Open'].values
# Getting the predicted prices
dataset_total = pd.concat((dataset_train[['Open','High','Low','Adj Close']], dataset_test[['Open','High','Low','Adj Close']]), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-4,4)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 111):
    X_test.append(inputs[i-60:i, :])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 4))
predicted_spy_price = regressor.predict(X_test)
#inputs.shape
df_zeros = np.zeros(shape=(len(predicted_spy_price), 4) )
df_zeros[:,0] = predicted_spy_price[:,0]
predicted_spy_price = sc.inverse_transform(df_zeros)[:,0]
# Visualising the results
plt.plot(real_spy_price, color = 'red', label = 'Real SPY Price')
plt.plot(predicted_spy_price, color = 'blue', label = 'Predicted SPY Price')
plt.title('SPY Price Prediction')
plt.xlabel('Time')
plt.ylabel('SPY Price')
plt.legend()
plt.show()
# import the dataset
dataset = pd.read_csv("../input/ftse-100-index/ftse.csv")
dataset.head()
#dataset.tail()
dataset['Adj Close'].mean()
dataset['Adj Close'].max()
dataset['Adj Close'].min()
dataset['Adj Close'].tail(60).mean()
dataset['Adj Close'].tail(60).max()
df = pd.read_csv("../input/ftse-100-index/ftse.csv",index_col='Date')
df.head()
import matplotlib.pyplot as plt
%matplotlib inline
df['Adj Close'].plot(figsize=(16,6))
plt.axhline(y=7395, color='g', linestyle='-')
plt.axhline(y=7779, color='r', linestyle='-')
plt.axhline(y=6889, color='r', linestyle='-')
dataset.shape
# append SPY Open in FTSE
spy_open = spy.iloc[:,0:2]
spy_open = spy_open.rename(columns={'Open':'Open_spy'})
spy_open.head()

dataset_new = dataset.merge(spy_open,on='Date',how='inner',left_index=True,right_index=False, indicator=True)
dataset_new.shape
dataset_new.head()
dataset_new.tail()
# split into train and test dataset
#dataset_train = dataset.iloc[0:270, 1:2]
dataset_train = dataset_new.iloc[0:200, 1:8]
#dataset_test = dataset.iloc[270:300,1:2]
dataset_test = dataset_new.iloc[200:248,1:8]
dataset_train.drop('Close',axis=1,inplace=True)
# train dataset
dataset_train.head()
dataset_train.shape
#dataset_train.tail()
dataset_test.drop('Close',axis=1,inplace=True)
# test dataset
dataset_test.head()
dataset_test.shape
#dataset_test.tail()
# convert train dataset into an array
training_set = dataset_train.values
#training_set
# perform feature scaling 
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 200):
    X_train.append(training_set_scaled[i-60:i, :])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 6))
# Building the RNN
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Masking
# Initialising the RNN
regressor = Sequential()
# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 6)))
regressor.add(Dropout(0.2))
# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))
# Adding the output layer
regressor.add(Dense(units = 1))
# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
# Making predictions and visualising the results
# Getting the real FTSE prices for the period 28/03/2018 to 11/05/2018
real_ftse_price = dataset_test['Open'].values
real_ftse_price
# Getting the predicted prices
dataset_total = pd.concat((dataset_train[['Open','High','Low','Adj Close','Volume','Open_spy']], dataset_test[['Open','High','Low','Adj Close','Volume','Open_spy']]), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-6,6)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 107):
    X_test.append(inputs[i-60:i, :])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 6))
predicted_ftse_price = regressor.predict(X_test)

predicted_ftse_price.shape
#inputs.shape
df_zeros = np.zeros(shape=(len(predicted_ftse_price), 6) )
df_zeros[:,0] = predicted_ftse_price[:,0]
#df_zeros
predicted_ftse_price = sc.inverse_transform(df_zeros)[:,0]
predicted_ftse_price
# Visualising the results
plt.plot(real_ftse_price, color = 'red', label = 'Real FTSE Price')
plt.plot(predicted_ftse_price, color = 'blue', label = 'Predicted FTSE Price')
plt.title('FTSE Price Prediction')
plt.xlabel('Time')
plt.ylabel('FTSE Price')
plt.legend()
plt.show()
# import the dataset
dataset = pd.read_csv("../input/ftse1predictions/ftse1.csv")
dataset.head()
dataset.tail()
dataset.shape
#dataset.fillna(-1,inplace=True)
dataset.tail()
dataset.drop(['Close','Volume'],axis=1,inplace=True)
dataset.head()
dataset.fillna(method='ffill',inplace=True)
#dataset.iloc[300:321,]
#dataset_new = dataset.merge(spy_open,on='Date',how='left',left_index=True,right_index=False, indicator=True)
#dataset_new.shape
#dataset_new.head()
#dataset_new.tail()
#dataset_new.fillna(method='ffill',inplace=True)
#dataset_new.tail()
# split into train and test datasets
dataset_train = dataset.iloc[0:310, 1:5]
dataset_train.tail()
#dataset_train.shape
dataset_test = dataset.iloc[310:321, 1:5]
dataset_test.tail()
#dataset_train.drop('Close',axis=1,inplace=True)
#dataset_train.head()
#dataset_train.fillna(method='bfill',inplace=True)
dataset_train.head()
dataset_train.tail()
dataset_train.corr()
#dataset_train.drop(['Volume','Open_spy'],axis=1,inplace=True)
#dataset_train.head()
# convert into array
training_set = dataset_train.values
# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
training_set_scaled
# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 310):
    X_train.append(training_set_scaled[i-60:i, :])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 4))
# Build and fit the RNN
# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
#regressor.add(Masking(mask_value=0, input_shape = (X_train.shape[1], 4)))
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 4)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
# test dataset
#dataset_test = dataset_new.iloc[270:320, 1:8]
#dataset_test.tail()
#dataset_test.shape
#dataset_test.head()
#dataset_test.drop(['Close','Volume','Open_spy'],axis=1,inplace=True)
#dataset_test.head()
real = dataset_test['Open'].values
#test
# Getting the predicted prices
dataset_total = pd.concat((dataset_train[['Open','High','Low','Adj Close']], dataset_test[['Open','High','Low','Adj Close']]), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-4,4)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 71):
    X_test.append(inputs[i-60:i, :])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 4))
predicted_ftse_price = regressor.predict(X_test)
#predicted_ftse_price = sc.inverse_transform(predicted_ftse_price)
#predicted_ftse_price
# invert transformation
df_zeros = np.zeros(shape=(len(predicted_ftse_price), 4) )
df_zeros[:,0] = predicted_ftse_price[:,0]
#df_zeros
predicted_ftse_price = sc.inverse_transform(df_zeros)[:,0]
predicted_ftse_price
# Visualising the results
plt.plot(real, color = 'red', label = 'Imputed FTSE Price')
plt.plot(predicted_ftse_price, color = 'blue', label = 'Predicted FTSE Price')
plt.title('FTSE Price Prediction')
plt.xlabel('Time')
plt.ylabel('FTSE Price')
plt.legend()
plt.show()
# import the GOLD dataset
dataset = pd.read_csv("../input/goldindex/gold.csv")
dataset.head()
#dataset.tail(20)
dataset.shape
dataset.fillna(method='ffill',inplace=True)
dataset.tail()
dataset_new = dataset.merge(spy_open,on='Date',how='left',left_index=True,right_index=False, indicator=True)
dataset_new.shape
dataset_new.head()
dataset_new.tail()
dataset_new.fillna(method='ffill',inplace=True)
dataset_new.tail()
dataset_new.head()
dataset_train = dataset_new.iloc[0:221,1:8]
dataset_train.head()
#dataset_train.shape
dataset_train.drop('Close',axis=1,inplace=True)
dataset_train.head()
dataset_train.corr()
dataset_train.drop(['Volume','Open_spy'],axis=1,inplace=True)
dataset_train.head()
training_set = dataset_train.values
# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 221):
    X_train.append(training_set_scaled[i-60:i, :])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 4))
# Build and fit the RNN
# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 4)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
dataset_test = dataset_new.iloc[221:271,1:8]
dataset_test.tail()
dataset_test.head()
dataset_test.shape
dataset_test.drop(['Close','Volume','Open_spy'],axis=1,inplace=True)
dataset_test.head()
# test values
test_values = dataset_test['Open'].values
# Getting the predicted prices
dataset_total = pd.concat((dataset_train[['Open','High','Low','Adj Close']], dataset_test[['Open','High','Low','Adj Close']]), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-4,4)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 110):
    X_test.append(inputs[i-60:i, :])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 4))
predicted_gold_price = regressor.predict(X_test)
#predicted_gold_price = sc.inverse_transform(predicted_gold_price)
#predicted_gold_price
# invert transformation
df_zeros = np.zeros(shape=(len(predicted_gold_price), 4) )
df_zeros[:,0] = predicted_gold_price[:,0]
predicted_gold_price = sc.inverse_transform(df_zeros)[:,0]
# Visualising the results
plt.plot(test_values, color = 'red', label = 'GOLD Price')
plt.plot(predicted_gold_price, color = 'blue', label = 'Predicted GOLD Price')
plt.title('GOLD Price Prediction')
plt.xlabel('Time')
plt.ylabel('GOLD Price')
plt.legend()
plt.show()