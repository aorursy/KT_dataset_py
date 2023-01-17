# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

import math

import sklearn

import keras

import sklearn.preprocessing

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder

import datetime

import os

import matplotlib.pyplot as plt

import tensorflow as tf

from sklearn.cluster import KMeans





from keras.layers import Dense

from keras.models import Sequential

from keras.utils import to_categorical

from keras.optimizers import SGD 

from keras.callbacks import EarlyStopping

from keras.utils import np_utils

import itertools

from keras.layers import LSTM

from keras.layers.convolutional import Conv1D

from keras.layers.convolutional import MaxPooling1D

from keras.layers import Dropout



from keras.utils import plot_model



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_loc = pd.read_csv('../input/air-quality-data-from-extensive-network-of-sensors/sensor_locations.csv')

df_loc.set_index('id', drop=True, inplace=True)

clusters = KMeans(n_clusters=9).fit(df_loc)

centroids = clusters.cluster_centers_

print(centroids)



plt.figure(figsize=(10,5))

plt.scatter(df_loc['latitude'], df_loc['longitude'], c= clusters.labels_.astype(float), cmap='viridis', s=50, alpha=0.5)

plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=20)

plt.xlabel('latitude')

plt.ylabel('longitude')

try:

    df_loc.reset_index(inplace=True)

except ValueError:

    pass

try:

    df_loc.drop(columns=['level_0', 'index'], inplace=True)

except KeyError:

    pass

arr = df_loc[['latitude', 'longitude']].values

df_loc['nearest_dist'] = 0

df_loc['nearest'] = df_loc.index.values

def fsi_numpy(item_id):

    ind = df_loc[df_loc.id == item_id].index.values[0]

    tmp_arr = arr - arr[ind]

    tmp_ser = np.sum( np.square( tmp_arr ), axis=1 )

    return tmp_ser

for item_id in df_loc.id.values:

    df_loc['dist'] = fsi_numpy(item_id)

    nearest5 = df_loc.copy().sort_values('dist').head(5)

    try:

        nearest5 = nearest5.drop(columns='level_0')

    except KeyError:

        pass

    nearest5 = nearest5.reset_index()

    df_loc.loc[df_loc[df_loc['id'] == item_id].index.values[0], 'nearest_dist'] = nearest5.dist.values[2]

    df_loc.loc[df_loc[df_loc['id'] == item_id].index.values[0], 'nearest'] = nearest5.id.values[2]

df_loc.sort_values('nearest_dist').head(5)
files = []

for r, d, f in os.walk("../input/air-quality-data-from-extensive-network-of-sensors"):

    for file in f:

        if '2017' in file:

            files.append(os.path.join(r, file))



sensor_num = '142'

df_jan = pd.read_csv(files[0])

df_sensor_col = [col for col in df_jan if col.startswith(sensor_num+'_') or col.startswith('UTC')]

df_sensor = pd.DataFrame()

for file in files:

    df_month = pd.read_csv(file)

    df_sensor = df_sensor.append(df_month[df_sensor_col])

    

df_sensor = df_sensor.dropna()



df_sensor['UTC time'] = pd.to_datetime(df_sensor['UTC time'])

df_sensor.set_index(df_sensor['UTC time'], inplace=True)
df = df_sensor.assign(date=df_sensor.index.date, time=df_sensor.index.time)

df.columns=['timestamp', 'temperature', 'humidity', 'pressure', 'pm1', 'pm25', 'pm10', 'date', 'time']

plt.figure(figsize=(20,10))

for date in df.date.unique():

    plt.plot('time', 'temperature', data=df[df.date == date].sort_values('time'))

plt.title('temperature daily')

plt.xlabel('time')

plt.ylabel('temperature (deg C)')

plt.show()

plt.close()


plt.figure(figsize=(20,10))

for date in df.date.unique():

    data = df[df.date == date].sort_values('time')

    plt.plot('time', 'pm25', data=data)

plt.title('pm25 daily')

plt.xlabel('time')

plt.ylabel('pm25')

plt.show()
def plot_all(df, title='Data', xlabel='Date', ylabel='Values'):

    plt.figure(figsize=(20,10))

    for col in df.columns[1:7]:

        print(col)

        plt.plot(df[col].sort_index())

    plt.legend(df.columns[1:7])

    plt.title(title)

    plt.xlabel(xlabel)

    plt.ylabel(ylabel)

plot_all(df, title='Before normalization')
def normalize_data(df):

    min_max_scaler = sklearn.preprocessing.MinMaxScaler()

    df['temperature'] = min_max_scaler.fit_transform(df.temperature.values.reshape(-1,1))

    df['humidity'] = min_max_scaler.fit_transform(df.humidity.values.reshape(-1,1))

    df['pressure'] = min_max_scaler.fit_transform(df.pressure.values.reshape(-1,1))

    df['pm1'] = min_max_scaler.fit_transform(df.pm1.values.reshape(-1,1))

    df['pm25'] = min_max_scaler.fit_transform(df.pm25.values.reshape(-1,1))

    df['pm10'] = min_max_scaler.fit_transform(df.pm10.values.reshape(-1,1))

    return df



normalized = normalize_data(df)

plot_all(normalized, title='After minmax scaler')
PM1 = 3

PM25 = 4

PM10 = 5



def prepare_data(df, ratio=0.8):

    for day in df.date.unique():

        daily_data = df[df.date==day]

        daily_numpy = daily_data.iloc[:, 1:7].values

        if day == df.date.unique()[0]:

            data = daily_numpy

        else:

            try:

                data = np.dstack([data, daily_numpy])

            except ValueError:

                print('Day {} does not match'.format(day))

    div = round(data.shape[2] * ratio)

    train = data[:, :, :div]

    test = data[:, :, div+1:]

    # 0:3 - input: temp, hum, press

    return train[:, :3, :], train[:, PM25, :], test[:, :3, :], test[:, PM25, :], div



train_x, train_y, test_x, test_y, div = prepare_data(normalized)

print('{} {} {} {}'.format(train_x.shape, train_y.shape, test_x.shape, test_y.shape))
def prepare_batch(train_x, train_y,batch_size, i):

    batch_x, batch_y = (train_x[:, :, i*batch_size:(i+1)*batch_size], train_y[:, i*batch_size:(i+1)*batch_size])

    print(batch_y.shape)

    print(batch_x.shape)

    batch_x = batch_x.reshape(batch_x.shape[2], batch_x.shape[0], batch_x.shape[1])

    batch_y = batch_y.reshape(batch_y.shape[0], batch_y.shape[1])

    return batch_x, batch_y
# design network

# x - [samples, timestamps, features]

# y - [samples]

train_Y = train_y.reshape(train_y.size)

test_Y = test_y.reshape(test_y.size)

train_X = train_x.transpose(0, 2, 1)

train_X = train_X.reshape(train_X.shape[0]*train_X.shape[1], 1, train_X.shape[2])

test_X = test_x.transpose(0, 2, 1)

test_X = test_X.reshape(test_X.shape[0]*test_X.shape[1], 1, test_X.shape[2])



train_X.shape, train_Y.shape, test_X.shape, test_Y.shape
# design network

model = Sequential()

model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]), batch_input_shape=(24, train_X.shape[1], train_X.shape[2]), return_sequences=True, stateful=True))

model.add(LSTM(50, return_sequences=False))

model.add(Dense(1))

model.compile(loss='mae', optimizer='adam', metrics=['mae', 'acc'])

# fit network

history = model.fit(train_X, train_Y, epochs=58, batch_size=24, validation_data=(test_X, test_Y), verbose=2, shuffle=False)
plt.figure(figsize=(10,5))

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('mae')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper right')

plt.show()

plot_model(model, show_shapes=True, show_layer_names=True, to_file='model.png')

from IPython.display import Image

Image(retina=True, filename='model.png')
n_steps = div-1 

batch_size = 3

n_inputs = train_x.shape[1]

n_neurons = 200 

n_outputs = 3

n_layers = 2

learning_rate = 0.001

n_epochs = train_x.shape[2]

train_set_size = train_x.shape[0]

test_set_size = test_x.shape[0]





tf.reset_default_graph()

#TODO:

X = tf.placeholder(tf.float32, [None, train_set_size, n_inputs])

y = tf.placeholder(tf.float32, [None, n_outputs])



layers = [tf.keras.layers.LSTMCell(units=n_neurons, activation=tf.nn.elu)

         for layer in range(n_layers)]

                                                                     

multi_layer_cell = tf.keras.layers.StackedRNNCells(layers)

output_layers = keras.layers.RNN(multi_layer_cell, return_state=True)

output = output_layers(X)



stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons]) 

stacked_outputs = keras.layers.dense(stacked_rnn_outputs, n_outputs)

outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])

last = outputs[:,n_steps-1,:] # keep only last output of sequence

#last = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)



weight = tf.Variable(tf.truncated_normal([n_neurons, int(y.get_shape()[1])]))

bias = tf.Variable(tf.constant(0.1, shape=[y.get_shape()[1]]))

last = tf.transpose(last, [1, 0])

prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)



loss = tf.reduce_mean(tf.square(last[-1] - y)) # loss function = mean squared error 

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) 

training_op = optimizer.minimize(loss)
with tf.Session() as sess: 

    sess.run(tf.global_variables_initializer())

    for i in range(int(n_epochs/batch_size)):

        print("Batch no {}".format(i))

        x_batch, y_batch = prepare_batch(train_x, train_y, batch_size, i)  

        sess.run(training_op, feed_dict={X: x_batch, y: y_batch}) # TODO: fix batches

        if epoch % 3 == 0:

            mse_train = loss.eval(feed_dict={X: x_train, y: y_train}) 

            mse_valid = loss.eval(feed_dict={X: x_valid, y: y_valid}) 

            print('%.2f epochs: MSE train/valid = %.6f/%.6f'%(

                iteration*batch_size/train_set_size, mse_train, mse_valid))



    y_train_pred = sess.run(outputs, feed_dict={X: x_train})

    y_valid_pred = sess.run(outputs, feed_dict={X: x_valid})

    y_test_pred = sess.run(outputs, feed_dict={X: x_test})
