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
train_X[0].shape
train_X = np.reshape(train_X,(-1, 28, 28))
fig=plt.figure(figsize=(8, 8))

for i in range(1, 11):

    fig.add_subplot(5, 5, i)

    plt.imshow(train_X[i])

plt.show()
print(train_Y[1:11].flatten())
train_X = np.array(train.drop("label",axis=1),dtype=np.float32)

train_Y = np.array(train[["label"]],dtype=np.int32)

train_X = np.reshape(train_X,(-1, 28, 28, 1))

val_X = np.reshape(val_X,(-1, 28, 28, 1))
all_x = np.concatenate((train_X, val_X)) / 255. * 2 - 1
train_X, val_X, train_Y, val_Y = train_test_split(all_x, np.concatenate((train_Y, val_Y)), shuffle=True)
from keras.layers import Conv2D, BatchNormalization, Dense, Flatten, Dropout
model = keras.Sequential()

model.add(Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu', input_shape=(28, 28, 1)))

model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same', activation='relu', input_shape=(28, 28, 1)))

model.add(BatchNormalization())

model.add(Conv2D(256, (1, 1), strides=(2, 2), padding='same', activation='relu', input_shape=(28, 28, 1)))

model.add(BatchNormalization())

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(BatchNormalization())

model.add(Dense(64, activation='relu'))

model.add(BatchNormalization())

model.add(Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop', loss="sparse_categorical_crossentropy", metrics=["accuracy"])
from keras.callbacks import ModelCheckpoint

filepath="best-weights.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
from tensorflow.keras.callbacks import LearningRateScheduler

scheduler = LearningRateScheduler(lambda e: 0.01 if e < 5 else 0.001 if e < 10 else 0.0001 if e < 15 else 0.00001 if e < 20 else 0.0000001)
from keras.callbacks import CSVLogger

csv_logger = CSVLogger('log.csv', append=True, separator=';')
callbacks = [checkpoint, scheduler, csv_logger]
model.fit(train_X, train_Y, epochs=25, validation_data=(val_X, val_Y), callbacks=callbacks)
model.load_weights("best-weights.hdf5")
print(np.argmax(model.predict(np.array([val_X[2]]))), val_Y[2][0])
test_csv = pd.read_csv("../input/Kannada-MNIST/test.csv")

X_test = np.array(test_csv.drop("id",axis=1),dtype=np.float32)

X_test = np.reshape(X_test,(-1, 28, 28, 1))

X_test = X_test / 255. * 2 - 1

results = model.predict(X_test)

results = np.argmax(results, axis=1)
results
submission = pd.read_csv("../input/Kannada-MNIST/sample_submission.csv")

submission['label'] = results

submission.to_csv("submission.csv",index=False)