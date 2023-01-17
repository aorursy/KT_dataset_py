# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%matplotlib inline

import matplotlib.pyplot as plt

import cv2

import numpy as np

from tensorflow import keras
train = np.loadtxt('/kaggle/input/digit-recognizer/train.csv', delimiter=',', skiprows=1)

test = np.loadtxt('/kaggle/input/digit-recognizer/test.csv', delimiter=',', skiprows=1)
# сохраняем разметку в отдельную переменную

train_label = train[:, 0]



# приводим размерность к удобному для обаботки виду

# добавляем размерность канала

train_img = np.resize(train[:, 1:], (train.shape[0], 28, 28, 1))

test_img = np.resize(test, (test.shape[0], 28, 28, 1))
fig = plt.figure(figsize=(20, 10))

for i, img in enumerate(train_img[0:5, :], 1):

    subplot = fig.add_subplot(1, 5, i)

    plt.imshow(img[:,:,0], cmap='gray');

    subplot.set_title('%s' % train_label[i - 1]);
from sklearn.model_selection import train_test_split

y_train, y_val, x_train, x_val = train_test_split(

    train_label, train_img, test_size=0.2, random_state=42)
seed = 123457

kernek_initializer = keras.initializers.glorot_normal(seed=seed)

bias_initializer = keras.initializers.normal(stddev=1., seed=seed)



model = keras.models.Sequential()



model.add(keras.layers.Conv2D(6, 

                              kernel_size=(5, 5), 

                              padding='same', 

                              activation='relu', 

                              input_shape=x_train.shape[1:],

                              bias_initializer=bias_initializer,

                              kernel_initializer=kernek_initializer))



model.add(keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'))



model.add(keras.layers.Conv2D(16, 

                              kernel_size=(5, 5),

                              padding='valid',

                              activation='relu', 

                              bias_initializer=bias_initializer,

                              kernel_initializer=kernek_initializer))



model.add(keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'))



model.add(keras.layers.Flatten())



model.add(keras.layers.Dense(32, activation='relu',

                             bias_initializer=bias_initializer,

                             kernel_initializer=kernek_initializer))



model.add(keras.layers.Dense(10, activation='softmax',

                             bias_initializer=bias_initializer,

                             kernel_initializer=kernek_initializer))



model.compile(optimizer='adam',

              loss='categorical_crossentropy',

              metrics=['accuracy'])
model.summary()
y_train_labels = keras.utils.to_categorical(y_train)
history = model.fit(x_train, 

          y_train_labels,

          batch_size=32, 

          epochs=5,

          validation_split=0.2)
import matplotlib.pyplot as plt

# Plot training & validation accuracy values

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
pred_val = model.predict_classes(x_val)
from sklearn.metrics import accuracy_score

print('Accuracy: %s' % accuracy_score(y_val, pred_val))
from sklearn.metrics import classification_report

print(classification_report(y_val, pred_val))
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_val, pred_val))
pred_test = model.predict_classes(test_img)
fig = plt.figure(figsize=(20, 10))

indices = np.random.choice(range(len(test_img)), 5)

img_prediction = zip(test_img[indices], pred_test[indices])

for i, (img, pred) in enumerate(img_prediction, 1):

    subplot = fig.add_subplot(1, 5, i)

    plt.imshow(img[...,0], cmap='gray');

    subplot.set_title('%d' % pred);
with open('submit.txt', 'w') as dst:

    dst.write('ImageId,Label\n')

    for i, p in enumerate(pred_test, 1):

        dst.write('%s,%d\n' % (i, p))
train = np.loadtxt('/kaggle/input/digit-recognizer/train.csv', delimiter=',', skiprows=1)

test = np.loadtxt('/kaggle/input/digit-recognizer/test.csv', delimiter=',', skiprows=1)
# сохраняем разметку в отдельную переменную

train_label = train[:, 0]



# приводим размерность к удобному для обаботки виду

# добавляем размерность канала

train_img = np.resize(train[:, 1:], (train.shape[0], 28, 28, 1))

test_img = np.resize(test, (test.shape[0], 28, 28, 1))
from sklearn.model_selection import train_test_split

y_train, y_val, x_train, x_val = train_test_split(

    train_label, train_img, test_size=0.2, random_state=42)
from __future__ import print_function

import keras

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras import backend as K



batch_size = 128

num_classes = 10

epochs = 12



# input image dimensions

img_rows, img_cols = 28, 28



'''# the data, split between train and test sets

(x_train, y_train), (x_test, y_test) = mnist.load_data()



if K.image_data_format() == 'channels_first':

    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)

    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)

    input_shape = (1, img_rows, img_cols)

else:

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)

    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    input_shape = (img_rows, img_cols, 1)'''

input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')

x_train /= 255

print('x_train shape:', x_train.shape)

print(x_train.shape[0], 'train samples')



# convert class vectors to binary class matrices

y_train = keras.utils.to_categorical(y_train, num_classes)



model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=input_shape))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adadelta(),

              metrics=['accuracy'])
model.summary()
history = model.fit(x_train, y_train,

          batch_size=32,

          epochs=12,

          verbose=1,

          validation_split=0.2)
import matplotlib.pyplot as plt

# Plot training & validation accuracy values

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
x_test = test_img.astype('float32')

x_test /= 255

print(x_test.shape[0], 'test samples')

#y_test = keras.utils.to_categorical(y_test, num_classes)



pred_test = model.predict_classes(x_test)
fig = plt.figure(figsize=(20, 10))

indices = np.random.choice(range(len(test_img)), 5)

img_prediction = zip(test_img[indices], pred_test[indices])

for i, (img, pred) in enumerate(img_prediction, 1):

    subplot = fig.add_subplot(1, 5, i)

    plt.imshow(img[...,0], cmap='gray');

    subplot.set_title('%d' % pred);
with open('submit.txt', 'w') as dst:

    dst.write('ImageId,Label\n')

    for i, p in enumerate(pred_test, 1):

        dst.write('%s,%d\n' % (i, p))