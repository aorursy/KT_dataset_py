# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from keras.datasets import fashion_mnist

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation

from keras.utils import np_utils



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

train_df = pd.read_csv("../input/fashion-mnist_train.csv")

test_df = pd.read_csv("../input/fashion-mnist_test.csv")
def show_img(img):

    plt.imshow(img,cmap='gray', interpolation='nearest')

    plt.show()
def separate_data(df):

#     results = []

    X_train = []

    Y_train = []

    for i in range(0, len(df.index)):

        row = df.iloc[i].values

#         X_train = X_train.reshape(28,28)

        Y_train.append(row[0])

        X_train = np.append(X_train, row[1:].reshape(28, 28) / 255)

#         X_train = X_train / 255



#         results.append(X_train)

    return X_train, Y_train
X_train = train_df.drop(columns=["label"], axis=1).values.reshape(60000, 28, 28, 1)

X_test = test_df.drop(columns=["label"], axis=1).values.reshape(10000, 28, 28, 1)
X_train = X_train / 255

X_test = X_test / 255
Y_train = train_df.label.values

Y_test = test_df.label.values
Y_train = np_utils.to_categorical(Y_train, 10)

Y_test = np_utils.to_categorical(Y_test, 10)
model = Sequential()



model.add(Activation('relu')) # An "activation" is just a non-linear function applied to the output

                              # of the layer above. Here, with a "rectified linear unit",

                              # we clamp all values below 0 to 0.

                           

model.add(Dropout(0.2))   # Dropout helps protect the model from memorizing or "overfitting" the training data

model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dropout(0.2))

model.add(Dense(10))

model.add(Activation('softmax')) 

from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.layers.core import Flatten
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

model.add(Dense(10))

model.add(Activation("softmax"))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

model.fit(X_train, Y_train, batch_size=128, epochs=4, verbose=1, validation_data=(X_test, Y_test))
model = Sequential()



model.add(Conv2D(32, (3, 3), padding="same", input_shape=(28, 28, 1)))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=chanDim))

model.add(Conv2D(32, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=chanDim))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



# second CONV => RELU => CONV => RELU => POOL layer set

model.add(Conv2D(64, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=chanDim))

model.add(Conv2D(64, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=chanDim))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



# first (and only) set of FC => RELU layers

model.add(Flatten())

model.add(Dense(512))

model.add(Activation("relu"))

model.add(BatchNormalization())

model.add(Dropout(0.5))



# softmax classifier

model.add(Dense(classes))

model.add(Activation("softmax"))

X_train, Y_train = separate_data(images)

X_test, Y_test = separate_data(dtest)

model.compile(loss='categorical_crossentropy', optimizer='adam')

model.fit(X_train, Y_train,

          batch_size=128, epochs=4, verbose=1,

          validation_data=(X_test, Y_test))