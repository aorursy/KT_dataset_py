#import packages



import argparse

import math

import sys

import time

import copy

import numpy as np

import pandas as pd



import keras

from keras.models import Sequential, Model

from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization, regularizers

from keras.layers.noise import GaussianNoise

from keras.layers import Conv2D, MaxPooling2D

from keras import backend as K

from keras.callbacks import ModelCheckpoint, EarlyStopping

from keras.utils.np_utils import to_categorical

K.set_image_dim_ordering('th')
#load data from csv files



data_train = np.genfromtxt('../input/fashion-mnist_train.csv', delimiter = ",", skip_header = 1)

data_test = np.genfromtxt('../input/fashion-mnist_test.csv', delimiter = ",", skip_header = 1)
#data extraction



X_train = data_train[:,1:]

y_train = data_train[:,0]

X_test = data_test[:,1:] = data_test[:,1:]
#data transformation



y_train = to_categorical(y_train) #convert to categories

num_classes = y_train.shape[1] #10 labels



X_train = X_train.astype('float32')

X_test = X_test.astype('float32')

X_train /= 255

X_test /= 255
#split validation and train dataset randomly



np.random.seed(12345)

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 

                                                  test_size=0.2, random_state=12345)
#reshape data into images



img_rows, img_cols = 28, 28

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)

X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

X_val = X_val.reshape(X_val.shape[0], 1, img_rows, img_cols)
#setting parameters



batch_size = 128

epochs = 10

filter_pixel=3

noise = 1

droprate=0.25

input_shape = (1, img_rows, img_cols)
#setting architecture

#best and fastest architecture for this challenge that I have tried



cnn3 = Sequential([

    Conv2D(64, kernel_size=(filter_pixel, filter_pixel), padding="same", activation='relu', 

           input_shape=input_shape),

    BatchNormalization(),

    Dropout(droprate),

    

    Conv2D(64, kernel_size=(filter_pixel, filter_pixel), activation='relu',border_mode="same"),

    BatchNormalization(),   

    MaxPooling2D(pool_size=(2, 2)),

    Dropout(droprate),

    

    Conv2D(64, kernel_size=(filter_pixel, filter_pixel), activation='relu',border_mode="same"),

    BatchNormalization(),

    MaxPooling2D(pool_size=(2, 2)),

    Dropout(droprate),

    

    Flatten(),

    Dense(512,use_bias=False),

    BatchNormalization(),

    Activation('relu'),

    Dropout(droprate),

    

    Dense(num_classes),

    Activation('softmax')

])
#compile the model



cnn3.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.RMSprop(),

              metrics=['accuracy'])
#first train



cnn3.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,

          validation_data=(X_val, y_val), shuffle=True)
#compile the model again. this helps combine effects of two methods



cnn3.compile(loss='sparse_categorical_crossentropy',

              optimizer=keras.optimizers.Adam(lr=0.0001),

              metrics=['accuracy'])
#prepare data for second train: for this loss and optimization, y needs not to be categorical



X_train = data_train[:,1:]

y_train = data_train[:,0]

X_test = data_test[:,1:] = data_test[:,1:]



X_train = X_train.astype('float32')

X_test = X_test.astype('float32')

X_train /= 255

X_test /= 255





X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)

X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

X_val = X_val.reshape(X_val.shape[0], 1, img_rows, img_cols)
#use another different random dataset for train and validation. Now use only 15% for validation 

#-> train with more data

np.random.seed(54321)

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 

                                                  test_size=0.15, random_state=54321)
#second train



cnn3.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,

          validation_data=(X_val, y_val), shuffle=True)
#use another different random dataset for train and validation. 

#This increases the randomness of the train. This time use 25% for validation 

#(as a test for model)



X_train = data_train[:,1:]

y_train = data_train[:,0]

X_test = data_test[:,1:] = data_test[:,1:]



X_train = X_train.astype('float32')

X_test = X_test.astype('float32')

X_train /= 255

X_test /= 255



X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)

X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

X_val = X_val.reshape(X_val.shape[0], 1, img_rows, img_cols)



np.random.seed(32145)

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 

                                                  test_size=0.25, random_state=32145)
#third train with same compile as second train



cnn3.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,

          validation_data=(X_val, y_val), shuffle=True, callbacks=callbacks)



#validation accuracy went up to 98% and stays.

#other trials show that score decrease after 3 training sessions. 

#Stop to avoid overfitting
#prepare new data split for data augmentation training. Use 20% of data for validation



X_train = data_train[:,1:]

y_train = data_train[:,0]

X_test = data_test[:,1:] = data_test[:,1:]



X_train = X_train.astype('float32')

X_test = X_test.astype('float32')

X_train /= 255

X_test /= 255



img_rows, img_cols = 28, 28

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)

X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

X_val = X_val.reshape(X_val.shape[0], 1, img_rows, img_cols)



np.random.seed(12345)

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 

                                                  test_size=0.2, random_state=12345)
#prepare for data augementation. Should check the images of test data first to determine the parameters

#for ImageDataGenerator because for large parametersthis will takes very large epoch to converge 



#this will add data and variation to the train, which increase the accuracy in predicting test data



from keras.preprocessing.image import ImageDataGenerator

batch_size = 128

gen = ImageDataGenerator(width_shift_range=0.05, shear_range=0.05, height_shift_range=0.05, 

                         zoom_range=0.02)

batches = gen.flow(X_train, y_train, batch_size=batch_size)

val_batches = gen.flow(X_val, y_val, batch_size=batch_size)
#train with data augmentation

#with a slight variation, needs about 40-50 epochs to get over 95% accuracy



cnn3.fit_generator(batches, steps_per_epoch=48000//batch_size, epochs=50, 

                    validation_data=val_batches, validation_steps=12000//batch_size, 

                   use_multiprocessing=False)
scores = model.evaluate(X_test, y_test, verbose=0)

print('Test loss:', scores[0])

print('Test accuracy:', scores[1])