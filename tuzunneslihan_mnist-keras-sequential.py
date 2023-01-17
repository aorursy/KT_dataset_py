from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Convolution2D, MaxPooling2D

from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense

from tensorflow.keras.optimizers import SGD

from keras.utils import np_utils

import tensorflow as tf

from tensorflow.keras.callbacks import TensorBoard

import pandas as pd

import csv

import numpy as np

import sys

import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'



img_rows, img_cols = 28, 28

img_channels = 1

num_pixels = img_cols*img_rows

num_classes = 10

num_trainImages = 42000

num_testImages = 28000



yTrain = np.ones((num_trainImages))

xTrain = np.ones((num_trainImages,img_channels,img_cols,img_rows))

counter = 0

skip = True



# train.csv should be in the same folder as this file

trainFile = open('../input/train.csv')

csv_file = csv.reader(trainFile)

for row in csv_file:

    if (skip == True):

        skip = False

        continue

    yTrain[counter] = row[0]

    temp = np.ones((1,num_pixels))

    for num in range(1,num_pixels):

        temp[0,num - 1] = row[num]

    temp = (temp - np.mean(temp))/(np.max(temp) - np.min(temp))

    temp = np.reshape(temp, (img_rows,img_cols))

    xTrain[counter,0,:,:] = temp

    counter += 1



yTest = np.ones((num_testImages))

xTest = np.ones((num_testImages,img_channels,img_cols,img_rows))

skip2 = True

counter2 = 0



testFile = open('../input/test.csv')

csv_file2 = csv.reader(testFile)

for row in csv_file2:

    if (skip2 == True):

        skip2 = False

        continue

    yTest[counter2] = row[0]

    temp = np.ones((1,num_pixels))

    for num in range(1,num_pixels):

        temp[0,num - 1] = row[num]

    temp = (temp - np.mean(temp))/(np.max(temp) - np.min(temp))

    temp = np.reshape(temp, (img_rows,img_cols))

    xTest[counter2,0,:,:] = temp

    counter2 += 1



# Convert class vectors to binary class matrices

yTrain = np_utils.to_categorical(yTrain, num_classes)

yTest = np_utils.to_categorical(yTest, num_classes)
# NETWORK ARCHITECTURE

model = Sequential()

model.add(Convolution2D(filters = 32, kernel_size = (3,3), padding = 'Same', input_shape=(img_channels,img_rows, img_cols)))

model.add(Activation('relu'))

model.add(Convolution2D(64,  kernel_size = (3,3), padding = 'Same'))

model.add(Activation('relu'))

model.add(Convolution2D(96,  kernel_size = (3,3), padding = 'Same'))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(1,1)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(64))

model.add(Activation('relu'))

model.add(Dropout(0.25))

model.add(Dense(num_classes))

model.add(Activation('softmax'))
# SPLIT THE DATASET

xTrain.shape

vTrain = xTrain[33600:42000,:,:,:]

vTest = yTrain[33600:42000,:]

xTrain = xTrain[:33600,:,:,:]

yTrain = yTrain[:33600,:]
# TRAINING

model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

model.fit(xTrain, yTrain, batch_size=32, nb_epoch=8,validation_data=(vTrain, vTest),shuffle=True)



results = np.zeros((num_testImages,2))

for num in range(1,num_testImages + 1):	

    results[num - 1,0] = num



# TESTING

temp = model.predict_classes(xTest, batch_size=32, verbose=1)

for num in range(0,num_testImages):	

    results[num,1] = temp[num]

# Results saved in this text file

np.savetxt('submission.csv', results, delimiter=',', fmt = '%i')  

results = pd.np.array(results)

firstRow = [[0 for x in range(2)] for x in range(1)]

firstRow[0][0] = 'ImageId'

firstRow[0][1] = 'Label'