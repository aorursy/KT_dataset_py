# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# read input train and test files
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
# separate the labels from train data
from keras.utils import np_utils

train_Y = np.array(train_df.label)
train_Y = np_utils.to_categorical(train_Y)
train_df.drop(columns=['label'], inplace=True)
print(train_Y.shape)
print(train_df.shape)
print(test_df.shape)
# df to np array conversion
train_X = np.array(train_df)
train_shape = train_X.shape

valid_X = train_X[int(train_shape[0]*0.75):]
valid_Y = train_Y[int(train_shape[0]*0.75):]

train_X = train_X[:int(train_shape[0]*0.75),:]
train_Y = train_Y[:int(train_shape[0]*0.75),:]
print(train_X.shape)
print(train_Y.shape)

print(valid_X.shape)
print(valid_Y.shape)

train_X_shape = train_X.shape
train_Y_shape = train_Y.shape

valid_X_shape = valid_X.shape
valid_Y_shape = valid_Y.shape
# reshape csv records into numpy images
train_X = train_X.reshape(train_X_shape[0], 28, 28, 1)
print(train_X.shape)

valid_X = valid_X.reshape(valid_X_shape[0], 28, 28, 1)
valid_X.shape
test_X = np.array(test_df)
print(test_X.shape)
test_X = test_X.reshape(test_X.shape[0], 28, 28,1)
test_X.shape
import numpy as np
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.applications import VGG16 
from keras.preprocessing import image
from keras.models import model_from_json
from keras import models,layers,optimizers
from keras.callbacks import ModelCheckpoint
from keras.applications.vgg16 import preprocess_input
from keras.layers import Flatten, MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.layers import Input, Conv2D, Dense, Reshape, Activation, Dropout
model = models.Sequential()

model.add(Convolution2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_X, train_Y, batch_size=256, epochs=100, verbose=1,validation_data=(valid_X,valid_Y))
 
test_predict = model.predict(test_X)
test_predict = np.argmax(test_predict,1)
test_predict = pd.Series(test_predict,name='Label')

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),test_predict],axis = 1)
submission.to_csv("Predictions.csv",index=False)
# predict_df = pd.DataFrame({'ImageId': list(range(1, len(labels)+1)), 'Label': labels})
# predict_df.to_csv('Predictions.csv')
# predict results
# results = model.predict(test)

# select the indix with the maximum probability
# results = np.argmax(results,axis = 1)

# results = pd.Series(results,name="Label")

# submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

# submission.to_csv("cnn_mnist_datagen.csv",index=False)
