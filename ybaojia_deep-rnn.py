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

trainX = train[:, 1:].reshape(train.shape[0], 28, 28)
trainX = trainX.astype(float)
trainX /= 255.0
import keras.utils.np_utils as kutils

trainY = kutils.to_categorical(train[:, 0])
class_num = trainY.shape[1]
print(class_num)
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.model_selection import train_test_split
random_seed = 7
np.random.seed(random_seed)

trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size = 0.2, random_state=random_seed)
model = Sequential()

model.add(LSTM(32, return_sequences=True, input_shape=(28, 28)))
model.add(Dropout(0.4))
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(256))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(class_num, activation='softmax'))

model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
model.summary()
#model.fit(trainX, trainY, batch_size=64, epochs=10, verbose=2)
hist = model.fit(trainX, trainY, batch_size=64, epochs=50, verbose=2, validation_data=(valX, valY))
testX = test.reshape(test.shape[0], 28, 28)
testX = testX.astype(float)
testX /= 255.0

yPred = model.predict_classes(testX)

np.savetxt('mnist-cnn.csv', np.c_[range(1,len(yPred)+1),yPred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')