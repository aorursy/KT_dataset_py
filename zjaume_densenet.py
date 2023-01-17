from __future__ import print_function

from keras.callbacks import LearningRateScheduler, ModelCheckpoint

from keras.models import Model, Sequential, load_model

from keras.layers import Input, Reshape, Dense, Dropout, Activation, Flatten, PReLU, Concatenate

from keras.layers import Conv2D, AveragePooling2D, GlobalAveragePooling2D

from keras.layers.normalization import BatchNormalization as BN

from keras.layers import GaussianNoise as GN

from keras.optimizers import SGD

import keras.backend as K

import numpy as np

import os

import tarfile

import sys

import pickle

print(os.listdir('../input/densenet200/'))

print(os.listdir("../input/cifar10/cifar-10-python/cifar-10-batches-py"))

import keras
def load_batch(fpath, label_key='labels'):

    """Internal utility for parsing CIFAR data.



    # Arguments

        fpath: path the file to parse.

        label_key: key for label data in the retrieve

            dictionary.



    # Returns

        A tuple `(data, labels)`.

    """

    with open(fpath, 'rb') as f:

        if sys.version_info < (3,):

            d = pickle.load(f)

        else:

            d = pickle.load(f, encoding='bytes')

            # decode utf8

            d_decoded = {}

            for k, v in d.items():

                d_decoded[k.decode('utf8')] = v

            d = d_decoded

    data = d['data']

    labels = d[label_key]



    data = data.reshape(data.shape[0], 3, 32, 32)

    return data, labels



train_num = 50000

x_train = np.zeros(shape=(train_num,3,32,32))

y_train = np.zeros(shape=(train_num))



test_num = 10000

x_test = np.zeros(shape=(test_num,3,32,32))

y_test = np.zeros(shape=(test_num))



def load_data():

    for i in range(1,6):

        begin = (i-1)*10000

        end = i*10000

        x_train[begin:end,:,:,:],y_train[begin:end] = load_batch("../input/cifar10/cifar-10-python/cifar-10-batches-py/data_batch_"+str(i))

    

    x_test[:],y_test[:] = load_batch("../input/cifar10/cifar-10-python/cifar-10-batches-py/test_batch")



load_data()

if K.image_data_format() == 'channels_last':

    x_test = x_test.transpose(0, 2, 3, 1)

    x_train = x_train.transpose(0, 2, 3, 1)
num_classes=10

#(x_train, y_train), (x_test, y_test) = cifar10.load_data()



x_train = x_train.astype('float32')

x_test = x_test.astype('float32')



x_train /= 255

x_test /= 255



print(x_train.shape)

print(x_test.shape)



y_train = keras.utils.to_categorical(y_train, num_classes)

y_test = keras.utils.to_categorical(y_test, num_classes)
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(

  featurewise_center=True,

  featurewise_std_normalization=True,

  width_shift_range=0.2,

  height_shift_range=0.2,

  rotation_range=20,

  zoom_range=[1.0,1.2],

  horizontal_flip=True)



datagen.fit(x_train)



testdatagen = ImageDataGenerator(

    featurewise_center=True,

    featurewise_std_normalization=True,

)



testdatagen.fit(x_train)
def bn_act(x, activation='relu'):

    l = BN()(x)

    if activation=='prelu':

        l = PReLU()(l)

    else:

        l = Activation('relu')(l)

    return l

        

def dense_block(x,N,k):

    concat = x

    for i in range(N):

        l = bn_act(concat)

        l = Conv2D(4*k, 1, strides=(1,1), padding='same', kernel_initializer='he_normal')(l)



        l = bn_act(l)

        l = Conv2D(k, 3, strides=(1,1), padding='same', kernel_initializer='he_normal')(l)

        concat = Concatenate()([concat,l])

    return concat



def transition(c):

    def inner(x):

        l = bn_act(x)

        convs = int(l.shape[3])*c

        l = Conv2D(int(convs), 1, strides=(1,1), padding='same', kernel_initializer='he_normal')(l)

        l = AveragePooling2D(2,strides=(2,2))(l)

        return l

    return inner



def densenet(input_shape):

    inpt = Input(shape = input_shape)



    k=12 # growth rate

    n=100 # depth or total number of layers

    N = (n-4)//6

    c = 0.5 # compression rate

    

    # stage 1

    x = Conv2D(k*2, 3, strides=(1,1), padding='same', kernel_initializer='he_normal')(inpt)

    

    # stage 2

    x = dense_block(x,N,k)

    x = transition(c)(x)

    

    # stage 3

    x = dense_block(x,N,k)

    x = transition(c)(x)



    # stage 4

    x = dense_block(x,N,k)

    

    x = bn_act(x)

    x = GlobalAveragePooling2D()(x)

    outpt = Dense(num_classes, activation="softmax")(x)

    model = Model(inpt,outpt)



    return model



def dense_scheduler(epoch):

    if epoch < 150:

        return 0.1

    if epoch < 225:

        return 0.01

    return 0.001



lrate = LearningRateScheduler(dense_scheduler)

chkpoint = ModelCheckpoint('densenet_40x12-best.hdf5', monitor='val_acc', save_best_only=True)



densenet = densenet(x_train.shape[1:])

densenet.summary()
batch_size=64

epochs=100



opt = SGD(lr=0.1, momentum=0.9, nesterov=True)

densenet.compile(loss='categorical_crossentropy',

              optimizer=opt,

              metrics=['accuracy'])

densenet = load_model('../input/densenet200/densenet-200')

history=densenet.fit_generator(datagen.flow(x_train, y_train,batch_size=batch_size),

                            steps_per_epoch=len(x_train) / batch_size,

                            epochs=epochs,

                            validation_data=testdatagen.flow(x_test, y_test),

                            validation_steps=len(x_test) / batch_size,

                            callbacks=[lrate,chkpoint],

                            verbose=2)
import matplotlib.pyplot as plt

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('DenseNet accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()