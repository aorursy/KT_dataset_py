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
#from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Average, LSTM
from sklearn.model_selection import train_test_split
random_seed = 7
np.random.seed(random_seed)

trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size = 0.2, random_state=random_seed)
from keras.models import Model

input_layer = Input(shape=(28, 28, 1), name='input')
cm1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
cm1 = Conv2D(64, (3, 3), activation='relu')(cm1)
cm1 = Dropout(0.5)(cm1)
cm1 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(cm1)
cm1 = Conv2D(128, (3, 3), padding='same', activation='relu')(cm1)
cm1 = Dropout(0.5)(cm1)
cm1 = MaxPooling2D(pool_size=(3, 3), strides=(3, 3))(cm1)
cm1 = Flatten()(cm1)
cm1 = Dense(128, activation='relu')(cm1)
cm1 = Dropout(0.5)(cm1)
cm1 = Dense(10, activation='softmax')(cm1)

cm2 = Conv2D(32, (5, 5), activation='relu')(input_layer)
cm2 = Conv2D(64, (5, 5), activation='relu')(cm2)
cm2 = Dropout(0.5)(cm2)
cm2 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(cm2)
cm2 = Conv2D(128, (3, 3), padding='same', activation='relu')(cm2)
cm2 = Dropout(0.5)(cm2)
cm2 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(cm2)
cm2 = Flatten()(cm2)
cm2 = Dense(128, activation='relu')(cm2)
cm2 = Dropout(0.5)(cm2)
cm2 = Dense(10, activation='softmax')(cm2)

output_layer = Average(name='output')([cm1, cm2])
model = Model(inputs=[input_layer], outputs=[output_layer])

model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
model.summary()
#model.fit(trainX, trainY, batch_size=128, epochs=20, verbose=2)
hist = model.fit(trainX, trainY, batch_size=64, epochs=500, verbose=2, validation_data=(valX, valY))
testX = test.reshape(test.shape[0], 28, 28, 1)
testX = testX.astype(float)
testX /= 255.0

y_prob = model.predict(testX)
yPred = y_prob.argmax(axis=-1)

np.savetxt('mnist-cnn.csv', np.c_[range(1,len(yPred)+1),yPred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')