# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import matplotlib.pyplot as plt



from keras.models import Sequential, load_model

from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
print(train.shape)

print(test.shape)
y_train = train['label']

x_train = train.drop('label',axis=1)

print(y_train.shape)

print(x_train.shape)
# PIXELS ARE FORMED INTO SET OF IMAGES OF SIZE 28*28*1



x_train = np.array(x_train).reshape(42000,28,28,1)

y_train = np.array(y_train)

x_test = np.array(test).reshape(28000,28,28,1)
x_train = x_train/255

x_test = x_test/255

print(x_train.shape)

print(x_test.shape)
model = Sequential()

model.add(Conv2D(16, kernel_size=4, input_shape=[28,28,1], activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(32, kernel_size=4, activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(2))

model.add(Conv2D(64, kernel_size=4, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.1))

model.add(Conv2D(128, kernel_size=4, activation='relu'))

model.add(BatchNormalization())

model.add(Flatten())

model.add(Dense(3200, activation='tanh'))

model.add(BatchNormalization())

model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x = x_train,y = y_train,epochs = 10)
pred = model.predict_classes(x_test)
demo = x_test[10]

plt.imshow(demo.reshape(28,28))
pred[10]