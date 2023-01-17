from __future__ import print_function

from keras.datasets import cifar10

from keras.models import Model, Sequential

from keras.layers import Input, Reshape, Dense, Dropout, Activation, Flatten, PReLU

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D

from keras.layers.normalization import BatchNormalization as BN

from keras.layers import GaussianNoise as GN

from keras.optimizers import SGD

import keras.backend as K

import numpy as np

import os

import tarfile

import sys

import pickle

print(os.listdir("../input/cifar-10-python/cifar-10-batches-py"))

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

        x_train[begin:end,:,:,:],y_train[begin:end] = load_batch("../input/cifar-10-python/cifar-10-batches-py/data_batch_"+str(i))

    

    x_test[:],y_test[:] = load_batch("../input/cifar-10-python/cifar-10-batches-py/test_batch")



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
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau

import math



# learning rate schedule

def step_decay(epoch,lr):

    initial_lrate = 0.1

    drop = 0.5

    epochs_drop = 20.0

    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))

    return lrate



def scheduler(epoch):

  if epoch < 25:

    return .1

  elif epoch < 50:

    return 0.01

  else:

    return 0.001

  

def nin_scheduler(epoch):

  if epoch <= 60:

      return 0.05

  if epoch <= 120:

      return 0.01

  if epoch <= 160:

      return 0.002

  return 0.0004



lrate = LearningRateScheduler(step_decay)

lrate_nin = LearningRateScheduler(nin_scheduler)

reduce_lr = ReduceLROnPlateau(monitor='val_loss',

                              factor=0.2,

                              min_lr=0.0001,

                              patience=4)
from keras.callbacks import ModelCheckpoint

chkpoint = ModelCheckpoint('wideresnet_28x10-best.hdf5', monitor='val_acc', save_best_only=True)
from keras.layers import Add, ZeroPadding2D



def bn_act(x, activation='relu'):

    l = BN()(x)

    if activation=='prelu':

        l = PReLU()(l)

    else:

        l = Activation('relu')(l)

    return l

        

def res_block(convs, identity=True, k=1):

    def inner(x):

        if not identity:

            strides = (2,2)

        else:

            strides = (1,1)

            

        act = bn_act(x)

        l = Conv2D(convs*k, 3, strides=strides, padding='same', kernel_initializer='he_normal')(act)

        

        l = bn_act(l)

        l = Dropout(0.5)(l)

        l = Conv2D(convs*k, 3, strides=(1,1), padding='same', kernel_initializer='he_normal')(l)

        

        if not identity or x.shape[3]!=convs*k:

            shortcut = Conv2D(convs*k, 1, strides=strides, padding='same', kernel_initializer='he_normal')(act)

            l = Add()([l,shortcut])

        else:

            l = Add()([l,x])

        return l

    return inner



def wide_resnet(input_shape):

    inpt = Input(shape = input_shape)



    # stage 1

    x = Conv2D(16, 3, strides=(1,1), padding='same', kernel_initializer='he_normal')(inpt)

    #x = bn_act(x)

    #x = MaxPooling2D(pool_size=(3,3),strides=2)(x)

    

    k=10 # widening factor

    n=28 # depth or total number of layers

    N = (n-4)//6

    # stage 2

    for i in range(N):

        x = res_block(16,k=k)(x)



    # stage 3

    x = res_block(32,False,k=k)(x)

    for i in range(1,N):

        x = res_block(32,k=k)(x)



    # stage 4

    x = res_block(64,False,k=k)(x)

    for i in range(1,N):

        x = res_block(64,k=k)(x)

    

    x = bn_act(x)

    x = GlobalAveragePooling2D()(x)

    outpt = Dense(num_classes, activation="softmax")(x)

    model = Model(inpt,outpt)



    return model





def res_scheduler(epoch):

    if epoch < 60:

        return 0.1

    if epoch < 120:

        return 0.02

    if epoch < 160:

        return 0.004

    return 0.0008



lrate_res = LearningRateScheduler(res_scheduler)



wideresnet = wide_resnet(x_train.shape[1:])

wideresnet.summary()
batch_size=128

epochs=200



opt = SGD(lr=0.1, momentum=0.9, nesterov=True)

wideresnet.compile(loss='categorical_crossentropy',

              optimizer=opt,

              metrics=['accuracy'])



history=wideresnet.fit_generator(datagen.flow(x_train, y_train,batch_size=batch_size),

                            steps_per_epoch=len(x_train) / batch_size, 

                            epochs=epochs,

                            validation_data=testdatagen.flow(x_test, y_test),

                            validation_steps=len(x_test) / batch_size,

                            callbacks=[lrate_res,chkpoint],

                            verbose=2)



# scores = model.evaluate(x_test, y_test, verbose=1)

#print('Test loss:', scores[0])

#print('Test accuracy:', scores[1])

import matplotlib.pyplot as plt

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('ResNet accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()