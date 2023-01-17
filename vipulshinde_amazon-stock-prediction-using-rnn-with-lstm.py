import os
# Importing the libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import math
import pandas_datareader as web
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.callbacks.callbacks import EarlyStopping,ModelCheckpoint
import seaborn as sns
sns.set_style('whitegrid')
# Importing the dataset of Amazon stock details from yahoo finance
df = web.DataReader('AMZN',data_source='yahoo',start='2010-01-01',end='2019-12-31')
df
# Visualizing the closing price history
plt.figure(figsize=(12,6))
plt.plot(df['Close'],linewidth=1,color='blue',label='AMZN')
plt.title('Closing stock price of Amazon Inc.',fontsize=18)
plt.xlabel('Date',fontsize=15)
plt.ylabel('Close Price',fontsize=15)
plt.legend(fontsize = 15)
plt.show()
# Creating training data which includes Close column only
dataset = df.iloc[:,3:4].values
print(dataset)
# Checking print(type(dataset))
print(dataset.shape)
# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
dataset = sc.fit_transform(dataset)
print(dataset)
# Creating the datastucture with 120 days and 1 output
X_train = []
y_train = []

for i in range(120,len(dataset)):
    X_train.append(dataset[i-120:i,0])
    y_train.append(dataset[i,0])

# Transforming to numpy array
X_train, y_train = np.array(X_train), np.array(y_train)
print(X_train.shape)
print(y_train.shape)
# Reshaping the input vector as per need of LSTM
X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
print(X_train.shape)
# Builing the RNN model

# Initialising the model
model = Sequential()

# Adding the 1st LSTM layer along with dropout regularization
model.add(LSTM(units=100, return_sequences=True, input_shape = (X_train.shape[1],1)))
model.add(Dropout(0.2))

# Adding the 2nd LSTM layer along with dropout regularization
model.add(LSTM(units=100, return_sequences=True))
model.add(Dropout(0.2))

# Adding the 3rd LSTM layer along with dropout regularization
model.add(LSTM(units=100, return_sequences=True))
model.add(Dropout(0.2))

# Adding the 4th LSTM layer along with dropout regularization
model.add(LSTM(units=100, return_sequences=True))
model.add(Dropout(0.2))

# Adding the 5th LSTM layer along with dropout regularization
model.add(LSTM(units=100, return_sequences=True))
model.add(Dropout(0.5))

# Adding the 6th LSTM layer along with dropout regularization
model.add(LSTM(units=100))
model.add(Dropout(0.5))

# Adding the output layer
model.add(Dense(units=1))

# Compiling the RNN model
model.compile(optimizer = RMSprop(lr=0.001),loss='mean_squared_error')
# Initializing the callback functions
es = EarlyStopping(monitor = 'loss', patience = 15, restore_best_weights=False)
# Fitting the model to training set
model.fit(X_train,y_train,epochs=20, batch_size=64)
# Getting the Real future stock price data
df_ = web.DataReader('AMZN',data_source='yahoo',start='2020-01-01',end='2020-01-30')
df_
print(df_.shape)
data_future = df_.iloc[:,3:4].values
data_future[:5]
# Getting the predicted future stock price
df_total = pd.concat((df['Close'],df_['Close']), axis=0)
inputs = df_total[len(df_total)-len(df_)-120:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
inputs.shape
# Creating the test datastucture with 90 days data
X_test = []

for i in range(120,140):
    X_test.append(inputs[i-120:i,0])

# Transforming to numpy array
X_test = np.array(X_test)
X_test.shape
# Reshaping X_test in order with required input for LSTM
X_test =  np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
data_predicted = model.predict(X_test)
data_predicted = sc.inverse_transform(data_predicted)
data_predicted
data_future
# Part-3 Visualizing the data
plt.figure(figsize=(12,6))
plt.plot(data_future, color = 'red', label = 'Real Amazon Stock Prize')
plt.plot(data_predicted, color = 'blue', label = 'Predicted Google Stock Prize')
plt.title('Prediction of Amazon Stock Prices')
plt.xlabel('Date')
plt.ylabel('Close Stock Prize')
plt.legend()
plt.show()
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(data_future,data_predicted))
rmse