# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Imports
import numpy as np
import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
%matplotlib inline

# Loading the data (it's preloaded in Keras)
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=1000)

print(x_train.shape)
print(x_test.shape)
print(x_train[0])
print(y_train[0])
# One-hot encoding the output into vector mode, each of length 1000
tokenizer = Tokenizer(num_words=1000)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
print(x_train[0])
#One-hot encoding the output
num_classes = 2
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(y_train.shape)
print(y_test.shape)
model = Sequential()
model.add(Dense(1024,activation='relu',input_dim=1000))
model.add(Dropout(0.5))
model.add(Dense(512,activation='relu',input_dim=1000))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()
# Compiling the model using categorical_crossentropy loss, and rmsprop optimizer.
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
# Running and evaluating the model
hist = model.fit(x_train, y_train,
          batch_size=16,
          epochs=10,
          validation_data=(x_test, y_test), 
          verbose=2)
score = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: ", score[1])
epoch = range(1,11)
y = hist.history['loss']
y_val = hist.history['val_loss']
plt.plot(epoch,y,epoch,y_val)
plt.show()
# detect and init the TPU
import tensorflow as tf
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)

# instantiate a distribution strategy
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
with tpu_strategy.scope():
    model_gpu = Sequential()
    model_gpu.add(Dense(1024,activation='relu',input_dim=1000))
    model_gpu.add(Dropout(0.5))
    model_gpu.add(Dense(512,activation='relu',input_dim=1000))
    model_gpu.add(Dropout(0.5))
    model_gpu.add(Dense(num_classes, activation='softmax'))
    model_gpu.summary()
    
    # Compiling the model using categorical_crossentropy loss, and rmsprop optimizer.
    model_gpu.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
# Running and evaluating the model
hist_gpu = model_gpu.fit(x_train, y_train,
          batch_size=16 * tpu_strategy.num_replicas_in_sync,
          epochs=10,
          validation_data=(x_test, y_test), 
          verbose=2)
score = model_gpu.evaluate(x_test, y_test, verbose=0)
print("Accuracy: ", score[1])
epoch = range(1,11)
y = hist_gpu.history['loss']
y_val = hist_gpu.history['val_loss']
plt.plot(epoch,y,epoch,y_val)
plt.show()
