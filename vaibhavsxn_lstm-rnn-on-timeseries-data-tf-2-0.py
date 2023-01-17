import pandas as pd

import numpy as np

%matplotlib inline

import matplotlib.pyplot as plt
x = np.linspace(0,50,501)

y = np.sin(x)
x
y
plt.plot(x,y)
df = pd.DataFrame(data=y,index=x,columns=['Sine'])
df
len(df)
test_percent = 0.1
len(df)*test_percent
test_point = np.round(len(df)*test_percent)
test_ind = int(len(df) - test_point)
test_ind
train = df.iloc[:test_ind]

test = df.iloc[test_ind:]
train
test
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# IGNORE WARNING ITS JUST CONVERTING TO FLOATS

# WE ONLY FIT TO TRAININ DATA, OTHERWISE WE ARE CHEATING ASSUMING INFO ABOUT TEST SET

scaler.fit(train)
scaled_train = scaler.transform(train)

scaled_test = scaler.transform(test)
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
# scaled_train
# define generator

length = 2 # Length of the output sequences (in number of timesteps)

batch_size = 1 #Number of timeseries samples in each batch

generator = TimeseriesGenerator(scaled_train, scaled_train, length=length, batch_size=batch_size)
len(scaled_train)
len(generator) # n_input = 2
# scaled_train
# What does the first batch look like?

X,y = generator[0]
print(f'Given the Array: \n{X.flatten()}')

print(f'Predict this y: \n {y}')
# Let's redefine to get 10 steps back and then predict the next step out

length = 10 # Length of the output sequences (in number of timesteps)

generator = TimeseriesGenerator(scaled_train, scaled_train, length=length, batch_size=1)
# What does the first batch look like?

X,y = generator[0]
print(f'Given the Array: \n{X.flatten()}')

print(f'Predict this y: \n {y}')
length = 50 # Length of the output sequences (in number of timesteps)

generator = TimeseriesGenerator(scaled_train, scaled_train, length=length, batch_size=1)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,LSTM,SimpleRNN
# We're only using one feature in our time series

n_features = 1
# define model

model = Sequential()



# Simple RNN layer

model.add(SimpleRNN(50,input_shape=(length, n_features)))



# Final Prediction

model.add(Dense(1))



model.compile(optimizer='adam', loss='mse')
model.summary()
# fit model

model.fit_generator(generator,epochs=5)
model.history.history.keys()
losses = pd.DataFrame(model.history.history)

losses.plot()
first_eval_batch = scaled_train[-length:]
first_eval_batch
first_eval_batch = first_eval_batch.reshape((1, length, n_features))
model.predict(first_eval_batch)
scaled_test[0]
test_predictions = []



first_eval_batch = scaled_train[-length:]

current_batch = first_eval_batch.reshape((1, length, n_features))
current_batch.shape
current_batch
np.append(current_batch[:,1:,:],[[[99]]],axis=1)
test_predictions = []



first_eval_batch = scaled_train[-length:]

current_batch = first_eval_batch.reshape((1, length, n_features))



for i in range(len(test)):

    

    # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])

    current_pred = model.predict(current_batch)[0]

    

    # store prediction

    test_predictions.append(current_pred) 

    

    # update batch to now include prediction and drop first value

    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
test_predictions
scaled_test
true_predictions = scaler.inverse_transform(test_predictions)
true_predictions
test
# IGNORE WARNINGS

test['Predictions'] = true_predictions
test
test.plot(figsize=(12,8))
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss',patience=2)
length = 49

generator = TimeseriesGenerator(scaled_train,scaled_train,

                               length=length,batch_size=1)





validation_generator = TimeseriesGenerator(scaled_test,scaled_test,

                                          length=length,batch_size=1)
# define model

model = Sequential()



# Simple RNN layer

model.add(LSTM(50,input_shape=(length, n_features)))



# Final Prediction

model.add(Dense(1))



model.compile(optimizer='adam', loss='mse')
model.fit_generator(generator,epochs=20,

                   validation_data=validation_generator,

                   callbacks=[early_stop])
test_predictions = []



first_eval_batch = scaled_train[-length:]

current_batch = first_eval_batch.reshape((1, length, n_features))



for i in range(len(test)):

    

    # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])

    current_pred = model.predict(current_batch)[0]

    

    # store prediction

    test_predictions.append(current_pred) 

    

    # update batch to now include prediction and drop first value

    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
# IGNORE WARNINGS

true_predictions = scaler.inverse_transform(test_predictions)

test['LSTM Predictions'] = true_predictions

test.plot(figsize=(12,8))
full_scaler = MinMaxScaler()

scaled_full_data = full_scaler.fit_transform(df)
length = 50 # Length of the output sequences (in number of timesteps)

generator = TimeseriesGenerator(scaled_full_data, scaled_full_data, length=length, batch_size=1)
model = Sequential()

model.add(LSTM(50, input_shape=(length, n_features)))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

model.fit_generator(generator,epochs=6) #6 because it stopped early last time at 6
forecast = []



first_eval_batch = scaled_full_data[-length:]

current_batch = first_eval_batch.reshape((1, length, n_features))



for i in range(len(test)):

    

    # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])

    current_pred = model.predict(current_batch)[0]

    

    # store prediction

    forecast.append(current_pred) 

    

    # update batch to now include prediction and drop first value

    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
forecast = scaler.inverse_transform(forecast)
# forecast
df
len(forecast)
50*0.1
forecast_index = np.arange(50.1,55.1,step=0.1)
len(forecast_index)
plt.plot(df.index,df['Sine'])

plt.plot(forecast_index,forecast)