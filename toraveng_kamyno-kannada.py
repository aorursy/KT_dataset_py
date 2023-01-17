import numpy as np

import pandas as pd

import os

import struct

import matplotlib.pyplot as plt

import keras

from keras.layers import *

from keras.models import Sequential, load_model

from keras.optimizers import *

from sklearn.preprocessing import MinMaxScaler

from keras.callbacks import CSVLogger, ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator
train = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

test = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')



valid_part = 10

test_id = test.id

test = test.drop('id', axis=1)

y_train = train.label

x_train = train.drop('label', axis=1)



train_size = int(x_train.shape[0] / valid_part * (valid_part - 1))

x_valid, y_valid = x_train[train_size:], y_train[train_size:]

x_train, y_train = x_train[:train_size], y_train[:train_size]
scaler = MinMaxScaler(feature_range=(-1, 1))



rows, cols = 28, 28

x_train = x_train.astype('float32')

test = test.astype('float32')



scaler.fit(x_train)

x_train = scaler.transform(x_train)

x_valid = scaler.transform(x_valid)

test = scaler.transform(test)

print(x_train.min(), x_train.max())



x_train = x_train.reshape(x_train.shape[0], rows, cols, 1)

test = test.reshape(test.shape[0], rows, cols, 1)

x_valid = x_valid.reshape(x_valid.shape[0], rows, cols, 1)



y_train = keras.utils.to_categorical(y_train, 10)

y_valid = keras.utils.to_categorical(y_valid, 10)
train_datagen = ImageDataGenerator(rotation_range = 10,

                                   shear_range = 0.1,

                                   width_shift_range = 0.25,

                                   height_shift_range = 0.25,

                                   zoom_range = 0.25,

                                   horizontal_flip = False)

epochs = 40

batch_size = 1024



model = Sequential()

model.add(Conv2D(64,

              kernel_size=(5, 5),

              input_shape=(28, 28, 1),

              padding='same'))  

model.add(LeakyReLU(alpha=0.1))

model.add(BatchNormalization(gamma_initializer='uniform', epsilon=0.0001, momentum=0.15))

model.add(Conv2D(64,

              kernel_size=(3, 3),

              padding='same'))

model.add(LeakyReLU(alpha=0.1))

model.add(BatchNormalization(gamma_initializer='uniform', epsilon=0.0001, momentum=0.15))

model.add(Conv2D(64,

              kernel_size=(3, 3),

              padding='same'))

model.add(LeakyReLU(alpha=0.1))

model.add(BatchNormalization(gamma_initializer='uniform', epsilon=0.0001, momentum=0.15))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))

model.add(Conv2D(128,

              kernel_size=(5, 5),

              padding='same'))

model.add(LeakyReLU(alpha=0.1))

model.add(BatchNormalization(gamma_initializer='uniform', epsilon=0.0001, momentum=0.15))

model.add(Conv2D(128,

              kernel_size=(3, 3),

              padding='same'))

model.add(LeakyReLU(alpha=0.1))

model.add(BatchNormalization(gamma_initializer='uniform', epsilon=0.0001, momentum=0.15))

model.add(Conv2D(128,

              kernel_size=(3, 3),

              padding='same'))

model.add(LeakyReLU(alpha=0.1))

model.add(BatchNormalization(gamma_initializer='uniform', epsilon=0.0001, momentum=0.15))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))

model.add(Conv2D(256,

              kernel_size=(5, 5),

              padding='same'))

model.add(LeakyReLU(alpha=0.1))

model.add(BatchNormalization(gamma_initializer='uniform', epsilon=0.0001, momentum=0.15))

model.add(Conv2D(256,

              kernel_size=(3, 3),

              padding='same'))

model.add(LeakyReLU(alpha=0.1))

model.add(BatchNormalization(gamma_initializer='uniform', epsilon=0.0001, momentum=0.15))

model.add(Conv2D(256,

              kernel_size=(3, 3),

              padding='same'))

model.add(LeakyReLU(alpha=0.1))

model.add(BatchNormalization(gamma_initializer='uniform', epsilon=0.0001, momentum=0.15))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(256))

model.add(LeakyReLU(alpha=0.1))

model.add(BatchNormalization(gamma_initializer='uniform', epsilon=0.0001, momentum=0.15))

model.add(Dense(10, activation='softmax'))



model.compile(loss='categorical_crossentropy',

            optimizer=Adam(lr=0.001),

            metrics=['accuracy'])



model.fit_generator(

  train_datagen.flow(x_train, y_train, batch_size=batch_size),

  steps_per_epoch=100,

  epochs=epochs,

  callbacks=[CSVLogger('log.csv'), ModelCheckpoint('model.h5', save_best_only=True)],

  validation_data=(x_valid, y_valid),

  validation_steps=50,  

  verbose=1)



model = load_model('model.h5')

label = model.predict(test)

label = np.argmax(label, axis=1)

df = pd.DataFrame({'id': test_id, 'label': label})

df.to_csv('submission.csv', index=False)