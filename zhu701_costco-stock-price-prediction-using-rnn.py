# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



# Importing the training set

dataset_train = pd.read_csv('../input/COST Train.csv')

print(dataset_train)
# Selecting closing prices

training_set = dataset_train.iloc[:,4:5].values



# Feature Scaling

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(training_set)
# Creating a data structure with 120 timesteps and 1 output



X_train = []

y_train = []

for i in range(120, 1258):

    X_train.append(training_set_scaled[i-120:i, 0])

    y_train.append(training_set_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)



# Reshaping

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout



# Initialising the RNN

regressor = Sequential()



# Adding four LSTM layer and Dropout regularisation

regressor.add(LSTM(units = 60, return_sequences = True, input_shape = (X_train.shape[1], 1)))

regressor.add(Dropout(0.2))



regressor.add(LSTM(units = 60, return_sequences = True))

regressor.add(Dropout(0.2))



regressor.add(LSTM(units = 60, return_sequences = True))

regressor.add(Dropout(0.2))



regressor.add(LSTM(units = 60))

regressor.add(Dropout(0.2))



# Adding the output layer

regressor.add(Dense(units = 1))



# Compiling the RNN

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')



# Fitting the RNN to the Training set

regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
dataset_test = pd.read_csv('../input/COST Test.csv')

print(dataset_test)
real_stock_price = dataset_test.iloc[:,4:5].values



# Combing the last 120 prices from train set with the test set 

dataset_total = pd.concat((dataset_train['Close'], dataset_test['Close']), axis = 0) 

inputs = dataset_total[len(dataset_total) - len(dataset_test) - 120:].values

inputs = inputs.reshape(-1,1)



# Applying the same data processing 

inputs = sc.transform(inputs) 

X_test = []

for i in range(120, 432):

    X_test.append(inputs[i-120:i, 0])

X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)

predicted_stock_price = sc.inverse_transform(predicted_stock_price)



compare = pd.DataFrame(predicted_stock_price).join(dataset_test['Close']) 

compare.rename(columns={0: "Predicted", "Close": "Actual"})
# Visualizing the results 

plt.plot(real_stock_price, color = 'red', label = 'Actual Costco Stock Price')

plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Costco Stock Price')

plt.title('Costco Stock Price Prediction')

plt.xlabel('Date')

plt.ylabel('Costco Stock Price')

plt.legend()

plt.show()
# Evaluating 

import math

from sklearn.metrics import mean_squared_error

rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))

print(rmse)
# Predicting the stock price in the next 120 days 



new = pd.DataFrame(columns=['Close'],index=[0])

newdays = pd.concat((dataset_test['Close'],new['Close']), axis = 0) 

newdays = newdays[len(newdays) - 1 - 120:].values

newdays = newdays.reshape(-1,1)

newdays = sc.transform(newdays) 

X_pred = []



for i in range(120, 121):

    X_pred.append(newdays[i-120:i, 0]) 

X_pred = np.array(X_pred)

X_pred = np.reshape(X_pred, (X_pred.shape[0], X_pred.shape[1], 1))

new_stock_price = regressor.predict(X_pred)

new_stock_price = sc.inverse_transform(new_stock_price) 



print("The price of the next trading day will be: $", new_stock_price)