

import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional

from tensorflow.keras.optimizers import SGD

import math

from sklearn.metrics import mean_squared_error



# Some functions to help out with

def plot_predictions(test,predicted,cur):

    plt.plot(test, color='red',label='Actual Stock Price')

    plt.plot(predicted, color='blue',label='Predicted Stock Price')

    plt.plot(cur, color='orange',label='Current Stock Price')

    plt.title('Stock Price Prediction')

    plt.xlabel('Time')

    plt.ylabel('Stock Price')

    plt.legend()

    plt.show()



def return_rmse(test,predicted):

    rmse = math.sqrt(mean_squared_error(test, predicted))

    print("The root mean squared error is {}.".format(rmse))

    
import pandas as pd

dataset = pd.read_csv("../input/msft.csv",index_col='Dates', parse_dates=['Dates'], dayfirst =True)



dataset['PREDICT'] = dataset['PX_LAST'].shift(-10) #trying to predict 10 days ahead, can change this to any days.

df=dataset.dropna()

df=df.iloc[:][:]

print(df.head())

print(df.describe())

size_data = len(df)

#split data_set

train_ratio=10

size_train = int(size_data*(1-train_ratio/100))

size_test = int(size_data*(train_ratio/100))



print('size of dataset:' , size_data)

print('size_train:', size_train)

print('size_train:', size_test)





# Checking for missing values

# select column

#Dates	PX_LAST	PX_VOLUME	VOLATILITY_90D	BEST_ANALYST_RATING



selected_column_list = ['PX_LAST'

                        #,'VOLATILITY_90D'

                        ,'BEST_ANALYST_RATING'

                        ,'PREDICT']



num_fields = len(selected_column_list)

num_training_fields = num_fields -1



df = df[selected_column_list]



df_training_set = df[selected_column_list][:size_train]

df_test_set = df[selected_column_list][size_train:]



training_set = df_training_set.values

test_set     = df_test_set.values



size_train = len(df_training_set)

size_test  = len(df_test_set)
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()



plt.clf()

plt.plot(df_training_set['PX_LAST'],color='red') 

plt.plot(df_test_set["PX_LAST"],color='blue')



plt.legend(['Training set','Test set'])

plt.title('MSFT stock price')

plt.show()



# Scaling the training set

sc = MinMaxScaler(feature_range=(0,1))

#fit the entire dataset

sc.fit(df) 



training_set_scaled = sc.transform(training_set)

test_set_scaled = sc.transform(test_set)

X_train = []

y_train = []

for i in range(60,len(training_set)):

    X_train.append(training_set_scaled[i-60:i,:-1])

    y_train.append(training_set_scaled[i,-1]) # last column is actual price to predict

X_train, y_train = np.array(X_train), np.array(y_train)



# Reshaping X_train for efficient modelling

X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],num_training_fields))





# The LSTM architecture

regressor = Sequential()

# First LSTM layer with Dropout regularisation

regressor.add(LSTM(units=20, return_sequences=True, input_shape=(X_train.shape[1],num_training_fields)))

regressor.add(Dropout(0.2))

# Second LSTM layer

regressor.add(LSTM(units=40, return_sequences=True))

regressor.add(Dropout(0.2))

# Third LSTM layer

regressor.add(LSTM(units=40, return_sequences=True))

regressor.add(Dropout(0.2))

# Fourth LSTM layer

regressor.add(LSTM(units=20))

regressor.add(Dropout(0.2))

# The output layer

regressor.add(Dense(units=1))



# Compiling the RNN

regressor.compile(optimizer='rmsprop',loss='mean_squared_error')

# Fitting to the training set

regressor.fit(X_train,y_train,epochs=10,batch_size=64)



# Now to get the test set ready in a similar way as the training set.

# The following has been done so forst 60 entires of test set have 60 previous values which is impossible to get unless we take the whole 

# 'High' attribute data for processing

#df = pd.concat((df_training_set, df_test_set),axis=0)





inputs = df.iloc[:][len(training_set)-60:].values



inputs = inputs.reshape(-1,num_fields)

inputs  = sc.transform(inputs)

# Preparing X_test and predicting the prices

X_test = []

y_test = []

for i in range(60,len(inputs)):

    X_test.append(inputs[i-60:i,:-1])

    y_test.append(inputs[i,-1])

X_test = np.array(X_test)

y_test = np.array(y_test)



X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],num_training_fields))



predicted_stock_price_sc = regressor.predict(X_test)



#compute inverse scale

# delete last column

predict_test_set_scaled = test_set_scaled[:,:-1]

#add last column with predicted value

predict_test_set_scaled = np.concatenate((predict_test_set_scaled,predicted_stock_price_sc), axis=1 )

predict_test_set = sc.inverse_transform(predict_test_set_scaled)







# Visualizing the results for LSTM

#test

#actual

#curent



plot_predictions(test_set[:,-1],predict_test_set[:,-1],test_set[:,0])



# Evaluating our model

return_rmse(test_set[:,-1],predict_test_set[:,-1])