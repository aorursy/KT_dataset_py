# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import Flatten

from keras.layers.convolutional import Convolution2D

from keras.layers.convolutional import MaxPooling2D

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

from keras import backend as K



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Read training and test data files

test = pd.read_csv("../input/emnist-balanced-test.csv").values

train  = pd.read_csv("../input/emnist-balanced-train.csv").values
# Reshape and normalize training data

trainX = train[:, 1:].reshape(train.shape[0],1,28, 28).astype( 'float32' )

X_train = trainX / 255.0

y_train = train[:,0]



testX = test[:, 1:].reshape(test.shape[0],1,28, 28).astype( 'float32' )

X_test = testX / 255.0

y_test = test[:,0]



print(X_train.shape)

print(X_test.shape)
from sklearn import preprocessing

lb = preprocessing.LabelBinarizer()

y_train = lb.fit_transform(y_train)

y_test = lb.fit_transform(y_test)
model = Sequential()

K.set_image_dim_ordering('th')

model.add(Convolution2D(30, 5, 5, border_mode= 'valid' , input_shape=(1, 28, 28),activation= 'relu' ))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(60, 3, 3, activation= 'relu' ))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(120, 3, 3, activation= 'relu' ))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(128, activation= 'relu' ))

model.add(Dense(50, activation= 'relu' ))

model.add(Dense(47, activation= 'softmax' ))

  # Compile model

model.compile(loss= 'categorical_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])
history = model.fit(X_train, y_train,

          epochs=100,

          batch_size=10000)
test_loss, test_acc = model.evaluate(X_test,y_test)
test_loss, test_acc
import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline



# Plot the loss and accuracy curves for training and validation 

fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['acc'], color='b', label="Training accuracy")

legend = ax[1].legend(loc='best', shadow=True)