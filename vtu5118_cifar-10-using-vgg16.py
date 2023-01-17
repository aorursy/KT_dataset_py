# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#------------------------------------------------------------------------------

# VGG16 ON CIFAR_10

#------------------------------------------------------------------------------

import numpy as np

from tensorflow.keras.applications.vgg16 import VGG16

import tensorflow.keras as k

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Dropout

from keras.utils.np_utils import to_categorical

from tensorflow.keras import optimizers

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import accuracy_score
#------------------------------------------------------------------------------

# Using VGG16 model, with weights pre-trained on ImageNet.

#------------------------------------------------------------------------------



vgg16_model = VGG16(weights='imagenet',

                    include_top=False, 

                    classes=10,

                    input_shape=(32,32,3)# input: 32x32 images with 3 channels -> (32, 32, 3) tensors.

                   )
#Define the sequential model and add th VGG's layers to it

model = Sequential()

for layer in vgg16_model.layers:

    model.add(layer)
#------------------------------------------------------------------------------

# Adding hiddens  and output layer to our model

#------------------------------------------------------------------------------



from tensorflow.keras.layers import Dense, Flatten, Dropout

model.add(Flatten())

model.add(Dense(512, activation='relu', name='hidden1'))

model.add(Dropout(0.4))

model.add(Dense(256, activation='relu', name='hidden2'))

model.add(Dropout(0.4))

model.add(Dense(10, activation='softmax', name='predictions'))



model.summary()
#------------------------------------------------------------------------------

#  Loading CIFAR10 data

#------------------------------------------------------------------------------



(X_train, y_train), (X_test, y_test) = k.datasets.cifar10.load_data()



print("******************")

print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
# Convert class vectors to binary class matrices using one hot encoding

y_train_ohe = to_categorical(y_train, num_classes = 10)

y_test_ohe = to_categorical(y_test, num_classes = 10)
# Data normalization

X_train = X_train.astype('float32')

X_test = X_test.astype('float32')

X_train  /= 255

X_test /= 255



print("******************")

print(X_train.shape)

print(y_train_ohe.shape)

print(X_test.shape)

print(y_test_ohe.shape)
X_val = X_train[40000:]

y_val = y_train_ohe[40000:]

print(X_val.shape)

print(y_val.shape)
X_train = X_train[:40000]

y_train_ohe = y_train_ohe[:40000]

print(X_train.shape)

print(y_train_ohe.shape)
#------------------------------------------------------------------------------

# TRAINING THE CNN ON THE TRAIN/VALIDATION DATA

#------------------------------------------------------------------------------



# initiate SGD optimizer

sgd = optimizers.SGD(lr=0.001, momentum=0.9)



# For a multi-class classification problem

model.compile(loss='categorical_crossentropy',optimizer= sgd,metrics=['accuracy'])





def lr_scheduler(epoch):

    return 0.001 * (0.5 ** (epoch // 20))

reduce_lr = LearningRateScheduler(lr_scheduler)



mc = ModelCheckpoint('./weights.h5', monitor='val_accuracy', save_best_only=True, mode='max')





# initialize the number of epochs and batch size

EPOCHS = 100

BS = 128



# construct the training image generator for data augmentation

aug = ImageDataGenerator(

    rotation_range=20, 

    zoom_range=0.15, 

    width_shift_range=0.2, 

    height_shift_range=0.2, 

    shear_range=0.15,

    horizontal_flip=True, 

    fill_mode="nearest")

 

# train the model

history = model.fit_generator(

    aug.flow(X_train,y_train_ohe, batch_size=BS),

    validation_data=(X_val,y_val),

    steps_per_epoch=len(X_train) // BS,

    epochs=EPOCHS,

    callbacks=[reduce_lr,mc])



#We load the best weights saved by the ModelCheckpoint

model.load_weights('./weights.h5')
train_loss, train_accuracy = model.evaluate_generator(aug.flow(X_train,y_train_ohe, batch_size=BS), 156)

print('Training loss: {}\nTraining accuracy: {}'.format(train_loss, train_accuracy))
val_loss, val_accuracy = model.evaluate(X_val, y_val)

print('Validation loss: {}\nValidation accuracy: {}'.format(val_loss, val_accuracy))
test_loss, test_accuracy = model.evaluate(X_test,y_test_ohe,)

print('Testing loss: {}\nTesting accuracy: {}'.format(test_loss, test_accuracy))