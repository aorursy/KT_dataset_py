import pandas as pd

import numpy as np

%matplotlib inline

import matplotlib.pyplot as plt
# IGNORE THE CONTENT OF THIS CELL

# import tensorflow as tf

# tf.compat.v1.disable_eager_execution()
df = pd.read_csv('/kaggle/input/energydata_complete.csv',index_col='date',infer_datetime_format=True)
df.head()
df.info()
df['Windspeed'].plot(figsize=(12,8))
df['Appliances'].plot(figsize=(12,8))
len(df)
df.head(3)
df.tail(5)
df.loc['2016-05-01':]
df = df.loc['2016-05-01':]
df = df.round(2)
len(df)
# How many rows per day? We know its every 10 min

24*60/10
test_days = 2
test_ind = test_days*144
test_ind
# Notice the minus sign in our indexing



train = df.iloc[:-test_ind]

test = df.iloc[-test_ind:]
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

length = 144 # Length of the output sequences (in number of timesteps)

batch_size = 1 #Number of timeseries samples in each batch

generator = TimeseriesGenerator(scaled_train, scaled_train, length=length, batch_size=batch_size)
len(scaled_train)
len(generator) 
# scaled_train
# What does the first batch look like?

X,y = generator[0]
print(f'Given the Array: \n{X.flatten()}')

print(f'Predict this y: \n {y}')
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,LSTM
scaled_train.shape
# define model

model = Sequential()



# Simple RNN layer

model.add(LSTM(100,input_shape=(length,scaled_train.shape[1])))



# Final Prediction (one neuron per feature)

model.add(Dense(scaled_train.shape[1]))



model.compile(optimizer='adam', loss='mse')
model.summary()
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss',patience=1)

validation_generator = TimeseriesGenerator(scaled_test,scaled_test, 

                                           length=length, batch_size=batch_size)
model.fit_generator(generator,epochs=10,

                    validation_data=validation_generator,

                   callbacks=[early_stop])
model.history.history.keys()
losses = pd.DataFrame(model.history.history)

losses.plot()
first_eval_batch = scaled_train[-length:]
first_eval_batch
first_eval_batch = first_eval_batch.reshape((1, length, scaled_train.shape[1]))
model.predict(first_eval_batch)
scaled_test[0]
n_features = scaled_train.shape[1]

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
true_predictions = pd.DataFrame(data=true_predictions,columns=test.columns)
true_predictions
from tensorflow.keras.models import load_model
model.save("multivariate.h5")