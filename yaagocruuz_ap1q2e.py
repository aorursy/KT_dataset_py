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
import keras
from keras import layers
from keras import models
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from os import listdir, makedirs
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras import optimizers, regularizers
from keras import losses
from keras.preprocessing import image
from keras.datasets import mnist

import warnings
warnings.filterwarnings('ignore')
RESOLUTION = 150
BATCH_SIZE = 20

#if you need data augmentation processing
#train_datagen = ImageDataGenerator(
        #rescale=1./255,
        #shear_range=0.2,
        #zoom_range=0.2,
        #horizontal_flip=True,
        #validation_split=0.3)

data_datagen_e = ImageDataGenerator(rescale=1./255, validation_split=0.15)

train_generator_e = data_datagen_e.flow_from_directory(
        "../input/the-simpsons-characters-dataset/simpsons_dataset/simpsons_dataset",
        classes=['homer_simpson', 'ned_flanders', 'moe_szyslak', 'lisa_simpson', 
                 'bart_simpson', 'marge_simpson', 'krusty_the_clown', 
                 'principal_skinner', 'charles_montgomery_burns', 'milhouse_van_houten'],
        target_size=(RESOLUTION, RESOLUTION),
        batch_size=BATCH_SIZE,
        class_mode='categorical', subset="training")

val_generator_e = data_datagen_e.flow_from_directory(
        "../input/the-simpsons-characters-dataset/simpsons_dataset/simpsons_dataset",
        classes=['homer_simpson', 'ned_flanders', 'moe_szyslak', 'lisa_simpson', 
                 'bart_simpson', 'marge_simpson', 'krusty_the_clown', 
                 'principal_skinner', 'charles_montgomery_burns', 'milhouse_van_houten'],
        target_size=(RESOLUTION, RESOLUTION),
        batch_size=BATCH_SIZE,
        class_mode='categorical', subset="validation")
conv_base = keras.applications.resnet50.ResNet50(include_top=False, input_shape=(150,150,3))
def resnet50_pretrained_model(model_resnet50, dropout_=False, regularizer_=False, regularizer_weight=0.001):
  
    model = Sequential()
    model.add(model_resnet50)
    model.add(Flatten())
    if dropout_:
        model.add(Dropout(0.5))
    if regularizer_:
        model.add(Dense(256, activation='relu', 
                        kernel_regularizer=regularizers.l1_l2(l1=regularizer_weight,
                                                            l2=regularizer_weight)))
    else:
        model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', 
                  optimizer=optimizers.adagrad(lr=0.001), metrics=['acc'])
    return model

modele = resnet50_pretrained_model(conv_base, dropout_=False, regularizer_=False, regularizer_weight=0.001)
historye = modele.fit_generator(
        train_generator_e,
        steps_per_epoch=(11745 // BATCH_SIZE),
        epochs=50,
        validation_data=val_generator_e,
        validation_steps=(2066 // BATCH_SIZE) 
    )
acc = historye.history['acc']
val_acc = historye.history['val_acc']
loss = historye.history['loss']
val_loss = historye.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
