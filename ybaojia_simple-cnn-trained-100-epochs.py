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
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
train_df.head(5)
print(train_df.shape)
print(test_df.shape)
train = train_df.values
test = test_df.values

trainX = train[:, 1:].reshape(train.shape[0], 28, 28, 1)
trainX = trainX.astype(float)
trainX /= 255.0
import keras.utils.np_utils as kutils

trainY = kutils.to_categorical(train[:, 0])
class_num = trainY.shape[1]
print(class_num)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
model = Sequential()

model.add(Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(32, (5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))

model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(256, (5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='sigmoid'))
model.add(Dense(class_num, activation='softmax'))

model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
model.fit(trainX, trainY, batch_size=64, epochs=100, verbose=2)
testX = test.reshape(test.shape[0], 28, 28, 1)
testX = testX.astype(float)
testX /= 255.0

yPred = model.predict_classes(testX)

np.savetxt('mnist-cnn.csv', np.c_[range(1,len(yPred)+1),yPred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')