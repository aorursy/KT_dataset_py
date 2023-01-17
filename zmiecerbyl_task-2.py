import keras

import numpy as np

import pandas as pd

import os

import random

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split
train = pd.read_csv("../input/Kannada-MNIST/train.csv")

validate = pd.read_csv("../input/Kannada-MNIST/Dig-MNIST.csv")
train_X = np.array(train.drop("label",axis=1),dtype=np.float32)

train_Y = np.array(train[["label"]],dtype=np.int32)

val_X = np.array(validate.drop("label",axis=1),dtype=np.float32)

val_Y = np.array(validate[["label"]],dtype=np.int32)

train_X = np.reshape(train_X,(-1, 28, 28, 1))

val_X = np.reshape(val_X,(-1, 28, 28, 1))
all_x = np.concatenate((train_X, val_X))  # / 255. * 2 - 1
train_X, val_X, train_Y, val_Y = train_test_split(all_x, np.concatenate((train_Y, val_Y)), shuffle=True)
from keras.layers import Conv2D, BatchNormalization, Dense, Flatten, Dropout, AveragePooling2D, Activation
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler

filepath="best-weights_1.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

csv_logger = CSVLogger('log_1.csv', append=True, separator=';')

callbacks = [checkpoint, csv_logger]
model = keras.Sequential()

model.add(Conv2D(6, (5, 5), strides=(1, 1), padding='same', activation='tanh', input_shape=(28, 28, 1)))

model.add(BatchNormalization())

model.add(AveragePooling2D(pool_size=(2, 2), strides=2, padding='same'))

model.add(Activation('tanh'))

model.add(Conv2D(16, (5, 5), strides=(1, 1), padding='same', activation='tanh'))

model.add(BatchNormalization())

model.add(AveragePooling2D(pool_size=(2, 2), strides=2, padding='same'))

model.add(Activation('tanh'))

model.add(Conv2D(120, (5, 5), strides=(1, 1), padding='same', activation='tanh'))

model.add(BatchNormalization())

model.add(Flatten())

model.add(Dense(84, activation='tanh'))

model.add(Dense(10, activation='softmax'))

model.compile(optimizer='sgd', loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_X, train_Y, batch_size=16, epochs=30, validation_data=(val_X, val_Y), callbacks=callbacks)
model = keras.Sequential()

model.add(Conv2D(6, (5, 5), strides=(1, 1), padding='same', activation='tanh', input_shape=(28, 28, 1)))

model.add(BatchNormalization())

model.add(AveragePooling2D(pool_size=(2, 2), strides=2, padding='same'))

model.add(Activation('tanh'))

model.add(Conv2D(16, (5, 5), strides=(1, 1), padding='same', activation='tanh'))

model.add(BatchNormalization())

model.add(AveragePooling2D(pool_size=(2, 2), strides=2, padding='same'))

model.add(Activation('tanh'))

model.add(Conv2D(120, (5, 5), strides=(1, 1), padding='same', activation='tanh'))

model.add(BatchNormalization())

model.add(Flatten())

model.add(Dense(84, activation='tanh'))

model.add(Dense(10, activation='softmax'))

model.compile(optimizer='sgd', loss="sparse_categorical_crossentropy", metrics=["accuracy"])
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler

filepath="best_weights_2.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

scheduler = LearningRateScheduler(lambda e: 0.01 if e < 5 else 0.001 if e < 10 else 0.0001 if e < 15 else 0.00001 if e < 20 else 0.0000001)

csv_logger = CSVLogger('log_2.csv', append=True, separator=';')

callbacks = [checkpoint, scheduler, csv_logger]
model.fit(train_X, train_Y, batch_size=16, epochs=30, validation_data=(val_X, val_Y), callbacks=callbacks)
model.load_weights("best-weights_1.hdf5")
test_csv = pd.read_csv("../input/Kannada-MNIST/test.csv")

X_test = np.array(test_csv.drop("id",axis=1),dtype=np.float32)

X_test = np.reshape(X_test,(-1, 28, 28, 1))

X_test = X_test

results = model.predict(X_test)

results = np.argmax(results, axis=1)
results
submission = pd.read_csv("../input/Kannada-MNIST/sample_submission.csv")

submission['label'] = results

submission.to_csv("submission.csv",index=False)
def create_model():

    model = keras.Sequential()

    model.add(Conv2D(6, (5, 5), strides=(1, 1), padding='same', activation='tanh', input_shape=(28, 28, 1)))

    model.add(AveragePooling2D(pool_size=(2, 2), strides=2, padding='same'))

    model.add(Activation('tanh'))

    model.add(Conv2D(16, (5, 5), strides=(1, 1), padding='same', activation='tanh'))

    model.add(AveragePooling2D(pool_size=(2, 2), strides=2, padding='same'))

    model.add(Activation('tanh'))

    model.add(Conv2D(120, (5, 5), strides=(1, 1), padding='same', activation='tanh'))

    model.add(Flatten())

    model.add(Dense(84, activation='tanh'))

    model.add(Dense(10, activation='softmax'))

    return model



def get_callbacks(index):

    filepath="best_weights_{}.hdf5".format(index)

    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    scheduler = LearningRateScheduler(lambda e: 0.01 if e < 5 else 0.001 if e < 10 else 0.0001 if e < 15 else 0.00001 if e < 20 else 0.0000001)

    csv_logger = CSVLogger('log_{}.csv'.format(index), append=True, separator=';')

    callbacks = [checkpoint, scheduler, csv_logger]

    return callbacks
from keras import optimizers
# model = create_model()

# model.compile(optimizer='RMSprop', loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# callbacks = get_callbacks(3)

# model.fit(train_X, train_Y, batch_size=16, epochs=30, validation_data=(val_X, val_Y), callbacks=callbacks)
# model = create_model()

# model.compile(optimizer='Adagrad', loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# callbacks = get_callbacks(4)

# model.fit(train_X, train_Y, batch_size=16, epochs=30, validation_data=(val_X, val_Y), callbacks=callbacks)
# model = create_model()

# model.compile(optimizer='Adadelta', loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# callbacks = get_callbacks(5)

# model.fit(train_X, train_Y, batch_size=16, epochs=30, validation_data=(val_X, val_Y), callbacks=callbacks)
# model = create_model()

# model.compile(optimizer='Adam', loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# callbacks = get_callbacks(6)

# model.fit(train_X, train_Y, batch_size=16, epochs=30, validation_data=(val_X, val_Y), callbacks=callbacks)
# model = create_model()

# model.compile(optimizer='Adamax', loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# callbacks = get_callbacks(7)

# model.fit(train_X, train_Y, batch_size=16, epochs=30, validation_data=(val_X, val_Y), callbacks=callbacks)
# model = create_model()

# model.compile(optimizer='Nadam', loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# callbacks = get_callbacks(8)

# model.fit(train_X, train_Y, batch_size=16, epochs=30, validation_data=(val_X, val_Y), callbacks=callbacks)
# model.load_weights("best_weights_4.hdf5")
# test_csv = pd.read_csv("../input/Kannada-MNIST/test.csv")

# X_test = np.array(test_csv.drop("id",axis=1),dtype=np.float32)

# X_test = np.reshape(X_test,(-1, 28, 28, 1))

# X_test = X_test

# results = model.predict(X_test)

# results = np.argmax(results, axis=1)
# results
# submission = pd.read_csv("../input/Kannada-MNIST/sample_submission.csv")

# submission['label'] = results

# submission.to_csv("submission.csv",index=False)