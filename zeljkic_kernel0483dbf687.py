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
df_train = pd.read_csv('../input/fashion-mnist_train.csv')

df_test = pd.read_csv('../input/fashion-mnist_test.csv')
# def separate_data(df):

#     result = None

#     X_train = []

#     Y_train = []

#     for i in range(0, len(df.index)):

#         row = df.iloc[i].values

#         Y_train.append(row[0])

#         X_train = row[1:]

#         X_train = X_train.reshape(28,28)

#         results.append(X_train)

#     return X_train, Y_train

# import matplotlib.pyplot as plt



# for im in results:

#     plt.imshow(im/255, interpolation='nearest', cmap='gray')

#     plt.show()
from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation

from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.layers.core import Flatten

from keras.utils import np_utils
model = Sequential()

model.add(Dense(512, input_shape=(784,)))

model.add(Activation('relu')) # An "activation" is just a non-linear function applied to the output

                              # of the layer above. Here, with a "rectified linear unit",

                              # we clamp all values below 0 to 0.

                           

model.add(Dropout(0.2))   # Dropout helps protect the model from memorizing or "overfitting" the training data

model.add(Dense(512))



model.add(Activation('relu'))

model.add(Dropout(0.2))

model.add(Dense(10))

model.add(Activation('softmax')) # This special "softmax" activation among other things,

                                 # ensures the output is a valid probaility distribution, that is

                                 # that its values are all non-negative and sum to 1.
X_train = df_train.drop(columns=["label"], axis=1).values.reshape(60000, 28, 28, 1)

X_test = df_test.drop(columns=["label"], axis=1).values.reshape(10000, 28, 28, 1)



X_train = X_train/255

X_test = X_test/255



Y_train = df_train.label.values

Y_test = df_test.label.values



Y_train = np_utils.to_categorical(Y_train, 10)

Y_test = np_utils.to_categorical(Y_test, 10)
model = Sequential()



model.add(Conv2D(32, (3, 3), padding="same", input_shape=(28, 28, 1)))

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())

model.add(Dense(512, activation="relu"))

model.add(Dropout(0.3))



# softmax classifier

model.add(Dense(10, activation="softmax"))





model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

model.fit(X_train, Y_train, batch_size=128, epochs=4, verbose=1, validation_data=(X_test, Y_test))
model = Sequential()



model.add(Conv2D(64, (3, 3), padding="same", input_shape=(28, 28, 1)))

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())

model.add(Dense(512, activation="relu"))

model.add(Dropout(0.3))



# softmax classifier

model.add(Dense(10, activation="softmax"))



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

model.fit(X_train, Y_train, batch_size=128, epochs=4, verbose=1, validation_data=(X_test, Y_test))