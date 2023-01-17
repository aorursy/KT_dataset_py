from __future__ import print_function

import numpy as np

import keras

import pickle

from scipy.io import loadmat



from keras.utils import np_utils

from keras.models import Sequential

from keras.layers import MaxPooling2D, Dense, Dropout, Flatten, Activation, Conv2D

from keras import backend as K



import matplotlib.pyplot as plt



import os

print(os.listdir("../input"))
def load_data(data_path, width=28, length=28):

    mat = loadmat(data_path)

    data = mat['dataset']

    

    x_train = data['train'][0,0]['images'][0,0]

    y_train = data['train'][0,0]['labels'][0,0]

    x_test = data['test'][0,0]['images'][0,0]

    y_test = data['test'][0,0]['labels'][0,0]





    x_train = np.array(x_train)

    y_train = np.array(y_train)

    x_test = np.array(x_test)

    y_test = np.array(y_test)



    x_train = x_train.reshape(x_train.shape[0], length, width, 1)

    x_test = x_test.reshape(x_test.shape[0], length, width, 1)

    y_train = y_train.reshape(y_train.shape[0], 1)

    y_test = y_test.reshape(y_test.shape[0], 1)



    input_shape = (length, width, 1)

    y_test = y_test-1

    y_train = y_train-1

    print(input_shape)

    x_train = x_train.astype('float32')/255

    x_test = x_test.astype('float32')/255



    print('x_train shape:', x_train.shape)

    print(x_train.shape[0], 'train samples')

    print(x_test.shape[0], 'test samples')

    

    return ((x_train, y_train), (x_test, y_test))
def train(train_data, width=28, length=28, batch_size=256, epochs=25, number_classes=26):

    (x_train, y_train), (x_test, y_test) = train_data

    

    y_train = np_utils.to_categorical(y_train, number_classes)

    y_test = np_utils.to_categorical(y_test, number_classes)

    input_shape = (length, width, 1)

    

    model = Sequential()

    model.add(Flatten(input_shape=input_shape))

    model.add(Dense(256, activation='relu'))

    model.add(Dropout(0.2))

    model.add(Dense(512, activation='relu'))

    model.add(Dropout(0.2))

    model.add(Dense(number_classes, activation='softmax'))



    model.compile(loss='categorical_crossentropy',

              optimizer='adadelta',

              metrics=['accuracy'])

    

    history = model.fit(x_train, y_train,

          batch_size=batch_size,

          epochs=epochs,

          verbose=1,

          validation_data=(x_test, y_test))

    

    model.summary()



    score = model.evaluate(x_test, y_test, verbose=0)

    

    print('Test loss:', score[0])

    print('Test accuracy:', score[1]*100,"%")

    return history

train_data = load_data('../input/emnistletters/emnist-letters.mat')
mlp = train(train_data)
# list all data in history

print(mlp.history.keys())

# summarize history for accuracy

plt.plot(mlp.history['accuracy'])

plt.plot(mlp.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(mlp.history['loss'])

plt.plot(mlp.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()