# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.io import loadmat

import keras

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Dropout, Flatten, Dense

from sklearn.model_selection import train_test_split

from keras.utils import np_utils

from keras import backend as K



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



img_rows = 28

img_cols = 28



# Any results you write to the current directory are saved as output.
mnist = loadmat("../input/mnist-original.mat")

# dict.keys(mnist)

x = mnist['data'].T

y = mnist['label'][0]

print(x.shape)

y.shape
from sklearn.model_selection import train_test_split



x_test, x_train, y_test, y_train = train_test_split(x, y, test_size=0.25, random_state=0)

print(x_test)

print(x_train)
# Preprocess data for training model

#reshaping

#this assumes our data format

#For 3D data, "channels_last" assumes (conv_dim1, conv_dim2, conv_dim3, channels) while 

#"channels_first" assumes (channels, conv_dim1, conv_dim2, conv_dim3).

if K.image_data_format() == 'channels_first':

  x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)

  x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)

  input_shape = (1, img_rows, img_cols)

else:

  x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)

  x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

  input_shape = (img_rows, img_cols, 1)

  

#more reshaping

x_train = x_train.astype('float32')

x_test = x_test.astype('float32')

print('x_train shape: ', x_train.shape)
# transform labels to vectors

num_category = 10

y_train = np_utils.to_categorical(y_train, num_category)

y_test = np_utils.to_categorical(y_test, num_category)

print(y_train.shape)

y_train[0]
model = Sequential()



def cnnNet(m, output_shape):

    m.add(Conv2D(30, kernel_size=3, activation='relu', input_shape=input_shape))

    m.add(MaxPooling2D(2,2))

    m.add(Conv2D(15, kernel_size=3, activation='relu'))

    m.add(MaxPooling2D(2,2))

    m.add(Dense(7, activation='relu')) # <7 stops working, but higher values do nothing

    m.add(Flatten())

    m.add(Dense(units = output_shape, activation='softmax')) # 'sigmoid'))

    

cnnNet(model, num_category)
model.summary()
#Adaptive learning rate (adaDelta) is a popular form of gradient descent rivaled only by adam and adagrad

#categorical ce since we have multiple classes (10)

model.compile(loss=keras.losses.categorical_crossentropy,

             optimizer=keras.optimizers.Adadelta(),

             metrics=['accuracy'])
# Train our model on training dataset

model_log = model.fit(x_train, y_train,

          batch_size=128, 

          epochs=30, # 10,

         verbose=1,

         validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss: ', score[0])

print('Test accuracy: ', score[1])
# Visualize training progress

import os

import matplotlib.pyplot as plt

# for displaying inline images in notebook

%matplotlib inline



fig = plt.figure()

plt.subplot(2,1,1)

plt.plot(model_log.history['acc'])

plt.plot(model_log.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='lower_right')



plt.subplot(2,1,2)

plt.plot(model_log.history['loss'])

plt.plot(model_log.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper right')



plt.tight_layout()

fig