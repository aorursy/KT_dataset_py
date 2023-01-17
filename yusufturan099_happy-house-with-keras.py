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
import h5py



def load_dataset():

    train_data = h5py.File('../input/happy-house/happy-house-dataset/train_happy.h5', 'r')

    x_train = np.array(train_data['train_set_x'][:])

    y_train = np.array(train_data['train_set_y'][:])

    

    

    

    test_data = h5py.File('../input/happy-house/happy-house-dataset/test_happy.h5', 'r')

    x_test = np.array(test_data['test_set_x'][:])

    y_test = np.array(test_data['test_set_y'][:])

    

    y_train = y_train.reshape((y_train.shape[0], 1))

    

    y_test = y_test.reshape((y_test.shape[0], 1))

    

    return x_train, y_train,  x_test, y_test
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig = load_dataset()



# Normalize image vectors

X_train = X_train_orig/255.

X_test = X_test_orig/255.



# Reshape

Y_train = Y_train_orig

Y_test = Y_test_orig



print ("number of training examples = " + str(X_train.shape[0]))

print ("number of test examples = " + str(X_test.shape[0]))

print ("X_train shape: " + str(X_train.shape))

print ("Y_train shape: " + str(Y_train.shape))

print ("X_test shape: " + str(X_test.shape))

print ("Y_test shape: " + str(Y_test.shape))
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D

from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D

from keras.models import Model
# GRADED FUNCTION: HappyModel



def HappyModel(input_shape):

  

    #Implementation of the HappyModel.

    

    X_input = Input(input_shape)

    X = ZeroPadding2D(padding=(8,8))(X_input)

    X = Conv2D(filters=8 , kernel_size=(5,5), strides=(2,2))(X)

    X = BatchNormalization(axis = 3)(X)

    X = Activation("relu")(X)

    X = MaxPooling2D((4, 4), name='max_pool')(X)

    X = ZeroPadding2D(padding=(8,8))(X_input)

    X = Conv2D(filters=16 , kernel_size=(3,3), strides=(3,3))(X)

    X = BatchNormalization(axis = 3)(X)

    X = Activation("relu")(X)

    X = MaxPooling2D((2, 2), name='max_pool')(X)

    X = Flatten()(X)

    X = Dense(1 , activation="sigmoid" )(X)

    

    model = Model(X_input, X)

    

    return model
model = HappyModel(X_train.shape[1:])      #CALL MODEL
model.compile(optimizer="ADAM", loss="binary_crossentropy" , metrics=["accuracy"] )    #COMPILE
model.fit(X_train,Y_train, epochs=100, batch_size=32)         #FIT  DATA
model.evaluate(X_test,Y_test)              #TEST THE MODEL