# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import tensorflow as tf

from keras import layers

from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D

from keras.models import Model, load_model

from keras.preprocessing import image

from keras.utils import layer_utils

from keras.datasets import mnist

from keras.utils.data_utils import get_file

from keras.applications.imagenet_utils import preprocess_input

# import pydot

# from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot

from keras.utils import plot_model

# from resnets_utils import *

from keras.initializers import glorot_uniform

import scipy.misc

from matplotlib.pyplot import imshow

%matplotlib inline



import keras.backend as K

K.set_image_data_format('channels_last')

K.set_learning_phase(1)

# GRADED FUNCTION: identity_block



def identity_block(X, f, filters, stage, block):

    """

    Implementation of the identity block as defined in Figure 4

    

    Arguments:

    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)

    f -- integer, specifying the shape of the middle CONV's window for the main path

    filters -- python list of integers, defining the number of filters in the CONV layers of the main path

    stage -- integer, used to name the layers, depending on their position in the network

    block -- string/character, used to name the layers, depending on their position in the network

    

    Returns:

    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)

    """

    

    # defining name basis

    conv_name_base = "res" + str(stage) + block + "_branch"

    bn_name_base   = "bn"  + str(stage) + block + "_branch"

    

    # Retrieve Filters

    F1, F2, F3 = filters

    

    # Save the input value. You'll need this later to add back to the main path. 

    X_shortcut = X

    

    # First component of main path

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding="valid", 

               name=conv_name_base+"2a", kernel_initializer=glorot_uniform(seed=0))(X)

    #valid mean no padding / glorot_uniform equal to Xaiver initialization - Steve 

    

    X = BatchNormalization(axis=3, name=bn_name_base + "2a")(X)

    X = Activation("relu")(X)

    ### START CODE HERE ###

    

    # Second component of main path (≈3 lines)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding="same",

               name=conv_name_base+"2b", kernel_initializer=glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis=3, name=bn_name_base+"2b")(X)

    X = Activation("relu")(X)

    # Third component of main path (≈2 lines)





    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding="valid",

               name=conv_name_base+"2c", kernel_initializer=glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis=3, name=bn_name_base+"2c")(X)

    

    X = Add()([X, X_shortcut])

    X = Activation("relu")(X)

    ### END CODE HERE ###

    

    return X

# GRADED FUNCTION: convolutional_block



def convolutional_block(X, f, filters, stage, block, s = 2):

    """

    Implementation of the convolutional block as defined in Figure 4

    

    Arguments:

    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)

    f -- integer, specifying the shape of the middle CONV's window for the main path

    filters -- python list of integers, defining the number of filters in the CONV layers of the main path

    stage -- integer, used to name the layers, depending on their position in the network

    block -- string/character, used to name the layers, depending on their position in the network

    s -- Integer, specifying the stride to be used

    

    Returns:

    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)

    """

    

    # defining name basis

    conv_name_base = 'res' + str(stage) + block + '_branch'

    bn_name_base = 'bn' + str(stage) + block + '_branch'

    

    # Retrieve Filters

    F1, F2, F3 = filters

    

    # Save the input value

    X_shortcut = X





    ##### MAIN PATH #####

    # First component of main path 

    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', padding='valid', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)

    X = Activation('relu')(X)

    

    ### START CODE HERE ###



    # Second component of main path (≈3 lines)

    X = Conv2D(F2, (f, f), strides = (1, 1), name = conv_name_base + '2b',padding='same', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)

    X = Activation('relu')(X)



    # Third component of main path (≈2 lines)

    X = Conv2D(F3, (1, 1), strides = (1, 1), name = conv_name_base + '2c',padding='valid', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)



    ##### SHORTCUT PATH #### (≈2 lines)

    X_shortcut = Conv2D(F3, (1, 1), strides = (s, s), name = conv_name_base + '1',padding='valid', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)

    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)



    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)

    X = layers.add([X, X_shortcut])

    X = Activation('relu')(X)

    

    ### END CODE HERE ###

    

    return X

# GRADED FUNCTION: ResNet50



def ResNet50(input_shape = (28, 28, 1), classes = 10):

    """

    Implementation of the popular ResNet50 the following architecture:

    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3

    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER



    Arguments:

    input_shape -- shape of the images of the dataset

    classes -- integer, number of classes



    Returns:

    model -- a Model() instance in Keras

    """

    

    # Define the input as a tensor with shape input_shape

    X_input = Input(input_shape)



    

    # Zero-Padding

    X = ZeroPadding2D((3, 3))(X_input)

    

    # Stage 1

    X = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), name="conv",

               kernel_initializer=glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis=3, name="bn_conv1")(X)

    X = Activation("relu")(X)

    X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(X)



    # Stage 2

    X = convolutional_block(X, f=3, filters=[28, 28, 256], stage=2, block="a", s=1)

    X = identity_block(X, f=3, filters=[28, 28, 256], stage=2, block="b")

    X = identity_block(X, f=3, filters=[28, 28, 256], stage=2, block="c")

    ### START CODE HERE ###



    # Stage 3 (≈4 lines)

    # The convolutional block uses three set of filters of size [128,128,512], "f" is 3, "s" is 2 and the block is "a".

    # The 3 identity blocks use three set of filters of size [128,128,512], "f" is 3 and the blocks are "b", "c" and "d".

    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block="a", s=1)

    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block="b")

    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block="c")

    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block="d")

    

    # Stage 4 (≈6 lines)

    # The convolutional block uses three set of filters of size [256, 256, 1024], "f" is 3, "s" is 2 and the block is "a".

    # The 5 identity blocks use three set of filters of size [256, 256, 1024], "f" is 3 and the blocks are "b", "c", "d", "e" and "f".

    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block="a", s=2)

    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="b")

    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="c")

    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="d")

    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="e")

    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="f")

    



    # Stage 5 (≈3 lines)

    # The convolutional block uses three set of filters of size [512, 512, 2048], "f" is 3, "s" is 2 and the block is "a".

    # The 2 identity blocks use three set of filters of size [256, 256, 2048], "f" is 3 and the blocks are "b" and "c".

    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block="a", s=2)

    X = identity_block(X, f=3, filters=[512, 512, 2048], stage=5, block="b")

    X = identity_block(X, f=3, filters=[512, 512, 2048], stage=5, block="c")

    

    # filters should be [256, 256, 2048], but it fail to be graded. Use [512, 512, 2048] to pass the grading

    



    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"

    # The 2D Average Pooling uses a window of shape (2,2) and its name is "avg_pool".

    X = AveragePooling2D(pool_size=(2, 2), padding="same")(X)

    

    ### END CODE HERE ###



    # output layer

    X = Flatten()(X)

    X = Dense(classes, activation="softmax", name="fc"+str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    

    

    # Create model

    model = Model(inputs=X_input, outputs=X, name="ResNet50")



    return model

model = ResNet50(input_shape=(28, 28, 1), classes=10)

model.summary()
import numpy as np

import keras

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)

x_test = x_test.reshape(10000, 784)

x_train = x_train.astype('float32')

x_test = x_test.astype('float32')

x_train /= 255

x_test /= 255

x_train = np.asarray(x_train).reshape(-1,28,28,1)

x_test = np.asarray(x_test).reshape(-1,28,28,1)

y_train = keras.utils.to_categorical(y_train, 10)

y_test = keras.utils.to_categorical(y_test, 10)

# print(x_train.shape[0], 'train samples')

# print(x_test.shape[0], 'test samples')

# print(np.asarray(x_train).reshape(-1,28,28,1).shape)

print(x_train.shape)

print(y_train.shape)

print(x_test.shape)
from keras.optimizers import Adam

from keras_preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard



PATH = "./model_resnet_mnist"



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



model.fit(x_train, y_train, epochs = 20, batch_size = 32)

history = model.fit(x_train, y_train,

                    batch_size=64,

                    epochs=20,

                    verbose=1,

                    validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])