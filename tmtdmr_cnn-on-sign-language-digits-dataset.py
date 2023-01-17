# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Import data

data_x = np.load("../input/sign-language-digits-dataset/Sign-language-digits-dataset/X.npy")

data_y = np.load("../input/sign-language-digits-dataset/Sign-language-digits-dataset/Y.npy")
# Visualize to samples

img_size = 64

plt.subplot(1, 3, 1)

plt.imshow(data_x[200].reshape(img_size, img_size))

plt.axis('off')

plt.subplot(1, 3, 2)

plt.imshow(data_x[800].reshape(img_size, img_size))

plt.axis('off')

plt.subplot(1, 3, 3)

plt.imshow(data_x[600].reshape(img_size, img_size))

plt.axis('off')

# Train-Test Split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=42)
# Size of elements of train_test_split methods

print("x train shape: {}".format(x_train.shape))

print("y train shape: {}".format(y_train.shape))

print("x test shape: {}".format(x_test.shape))

print("y test shape: {}".format(y_test.shape))
# Reshaping. We reshape x_train and x_test because Keras requires 3 dimention.

x_train = x_train.reshape(-1,64,64,1)

x_test = x_test.reshape(-1,64,64,1)
# New size of x_train and x_shape

print("x train shape: {}".format(x_train.shape))

print("x test shape: {}".format(x_test.shape))
# Creating Model

from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop,Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

# building of our model

model = Sequential()



# we add convolutional layer, count of filter = 64, kernel_size means that dimension of filter.

model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (64,64,1))) 

# dimension of (64,64,1) is 3 because kernel requires 3 dimensions. Number "1" shows that it is used as gray scale. 

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25)) 



# we rewrite the top one. We don't have to write input shape because these are things that are connected to each other like chains.

model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 128, kernel_size = (3,3), padding = 'Same', activation = 'relu'))

model.add(MaxPool2D(pool_size = (2,2)))

model.add(Dropout(0.25))





# fully connected

model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dense(512, activation = 'relu'))

model.add(Dense(256, activation = "relu"))

model.add(Dense(10, activation = 'softmax')) 

# although sigma function is used for binary classification, softmax is a version of sigma function which is used for multi-output classification.
model.summary()
# defining optimizer

optimizer = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999)
# compiling optimizer

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
# fitting

history = model.fit(x_train,y_train,epochs=100,validation_data=(x_test,y_test))
scores = model.evaluate(x_test, y_test, verbose=0)

print("{}: {:.2f}%".format("accuracy", scores[1]*100))