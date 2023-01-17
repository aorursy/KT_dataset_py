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
DATA_SET = "/kaggle/input/power-consumption/household_power_consumption.txt"
# load all data

dataset = pd.read_csv(DATA_SET, sep=';', header=0, low_memory=False, infer_datetime_format=True, parse_dates={'datetime':[0,1]}, index_col=['datetime'])
dataset.head()
# mark all missing values

dataset.replace('?', np.nan, inplace=True)

# make dataset numeric

dataset = dataset.astype('float32')
# fill missing values with a value at the same time one day ago

def fill_missing(values):

	one_day = 60 * 24

	for row in range(values.shape[0]):

		for col in range(values.shape[1]):

			if np.isnan(values[row, col]):

				values[row, col] = values[row - one_day, col]
# fill missing

fill_missing(dataset.values)
dataset.head()
# save updated dataset

OUTPUT_CSV = "/kaggle/working/household_power_consumption.csv"

dataset.to_csv(OUTPUT_CSV)
dataset.info()
dataset = dataset[:10801]
dataset
train, test = dataset[1:-2880],dataset[-2880:-24]
train.shape,test.shape
train,test = train.iloc[:, 2:3].values,test.iloc[:, 2:3].values
len(train),test
# Feature Scaling

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(train)
len(training_set_scaled)
# Creating a data structure with 60 timesteps and 1 output

X_train = []

y_train = []

for i in range(60, 7920):

    X_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i, 0])

#print(X_train,y_train)

X_train, y_train = np.array(X_train), np.array(y_train)
X_train.shape
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_train.shape
# Importing the Keras libraries and packages

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout
# Initialising the RNN

regressor = Sequential()



# Adding the first LSTM layer and some Dropout regularisation

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))

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

regressor.fit(X_train, y_train, epochs = 25, batch_size = 42)
#original data

inputs = dataset["Voltage"]
inputs = inputs[len(inputs) - len(test) - 60:].values
inputs.shape
inputs = inputs.reshape(-1,1)

inputs = sc.transform(inputs)
X_test = []

for i in range(60, 2916):

    X_test.append(inputs[i-60:i, 0])

X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
X_test.shape
predicted_voltage = regressor.predict(X_test)

predicted_voltage = sc.inverse_transform(predicted_voltage)
len(predicted_voltage)
%matplotlib inline

import matplotlib.pyplot as plt



plt.plot(test, color = 'red', label = 'Real Voltage')

plt.plot(predicted_voltage, color = 'blue', label = 'Predicted voltage')

plt.title('Power consumption')

plt.xlabel('Time')

plt.ylabel('Voltage')

plt.legend()

plt.show()
#mse

from sklearn.metrics import mean_squared_error

import math



mse = mean_squared_error(test,predicted_voltage)



rmse = math.sqrt(mse)



print(f'MSE = {mse} and RMSE = {rmse}')
#plotting for less predictions

test.shape
#predecting for next 10 mins

test_less = test[:10]
test_less
inputs_less = dataset["Voltage"]

inputs_less = inputs_less[len(inputs_less) - len(test_less) - 60:].values
inputs_less = inputs_less.reshape(-1,1)

inputs_less = sc.transform(inputs_less)
inputs_less.shape
X_test_less = []

for i in range(60, inputs_less.shape[0]):

    X_test_less.append(inputs_less[i-60:i, 0])

X_test_less = np.array(X_test_less)

X_test_less = np.reshape(X_test_less, (X_test_less.shape[0], X_test_less.shape[1], 1))
predicted_voltage_less = regressor.predict(X_test_less)

predicted_voltage_less = sc.inverse_transform(predicted_voltage_less)
plt.plot(test_less, color = 'red', label = 'Real Voltage')

plt.plot(predicted_voltage_less, color = 'blue', label = 'Predicted voltage')

plt.title('Power consumption for next 10 mins')

plt.xlabel('Time')

plt.ylabel('Voltage')

plt.legend()

plt.show()
#mse

from sklearn.metrics import mean_squared_error

import math



mse = mean_squared_error(test_less,predicted_voltage_less)



rmse = math.sqrt(mse)



print(f'MSE = {mse} and RMSE = {rmse}')

from sklearn.externals import joblib 



OUTPUT_PKL = "/kaggle/working/power_60.pkl"



# Save the model as a pickle in a file 

joblib.dump(regressor, OUTPUT_PKL) 

  

# Load the model from the file 

knn_from_joblib = joblib.load(OUTPUT_PKL)  

  

# Use the loaded model to make predictions 

predicted_voltage_less = knn_from_joblib.predict(X_test_less)

predicted_voltage_less = sc.inverse_transform(predicted_voltage_less)
predicted_voltage_less
#training other model by considering 120 and predicting 121 and training model

#Creating a data structure with 120 timesteps and 1 output

X_train = []

y_train = []

for i in range(120, 7920):

    X_train.append(training_set_scaled[i-120:i, 0])

    y_train.append(training_set_scaled[i, 0])

#print(X_train,y_train)

X_train, y_train = np.array(X_train), np.array(y_train)



#reshape

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



# Initialising the RNN

regressor = Sequential()



# Adding the first LSTM layer and some Dropout regularisation

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))

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

regressor.fit(X_train, y_train, epochs = 25, batch_size = 42)
#predecting for next 10 mins

test_less = test[:10]

inputs_less = dataset["Voltage"]

inputs_less = inputs_less[len(inputs_less) - len(test_less) - 120:].values



inputs_less = inputs_less.reshape(-1,1)

inputs_less = sc.transform(inputs_less)



X_test_less = []

for i in range(120, inputs_less.shape[0]):

    X_test_less.append(inputs_less[i-120:i, 0])

X_test_less = np.array(X_test_less)

X_test_less = np.reshape(X_test_less, (X_test_less.shape[0], X_test_less.shape[1], 1))



predicted_voltage_less = regressor.predict(X_test_less)

predicted_voltage_less = sc.inverse_transform(predicted_voltage_less)



plt.plot(test_less, color = 'red', label = 'Real Voltage')

plt.plot(predicted_voltage_less, color = 'blue', label = 'Predicted voltage')

plt.title('Power consumption for next 10 mins')

plt.xlabel('Time')

plt.ylabel('Voltage')

plt.legend()

plt.show()



#mse

mse = mean_squared_error(test_less,predicted_voltage_less)



rmse = math.sqrt(mse)



print(f'MSE = {mse} and RMSE = {rmse}')

predicted_voltage_less
#writing to pkl file

OUTPUT_PKL = "/kaggle/working/power_120.pkl"



# Save the model as a pickle in a file 

joblib.dump(regressor, OUTPUT_PKL) 

  

# Load the model from the file 

knn_from_joblib = joblib.load(OUTPUT_PKL)  

  

# Use the loaded model to make predictions 

predicted_voltage_less = knn_from_joblib.predict(X_test_less)

predicted_voltage_less = sc.inverse_transform(predicted_voltage_less)
predicted_voltage_less