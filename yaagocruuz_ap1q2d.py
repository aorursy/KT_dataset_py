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
RESOLUTION = 64
BATCH_SIZE = 16

#if you need data augmentation processing
#train_datagen = ImageDataGenerator(
        #rescale=1./255,
        #shear_range=0.2,
        #zoom_range=0.2,
        #horizontal_flip=True,
        #validation_split=0.3)

data_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.15)

train_generator = data_datagen.flow_from_directory(
        "../input/simpsons_dataset/simpsons_dataset",
        classes=['homer_simpson', 'ned_flanders', 'moe_szyslak', 'lisa_simpson', 
                 'bart_simpson', 'marge_simpson', 'krusty_the_clown', 
                 'principal_skinner', 'charles_montgomery_burns', 'milhouse_van_houten'],
        target_size=(RESOLUTION, RESOLUTION),
        batch_size=BATCH_SIZE,
        class_mode='categorical', subset="training")

val_generator = data_datagen.flow_from_directory(
        "../input/simpsons_dataset/simpsons_dataset",
        classes=['homer_simpson', 'ned_flanders', 'moe_szyslak', 'lisa_simpson', 
                 'bart_simpson', 'marge_simpson', 'krusty_the_clown', 
                 'principal_skinner', 'charles_montgomery_burns', 'milhouse_van_houten'],
        target_size=(RESOLUTION, RESOLUTION),
        batch_size=BATCH_SIZE,
        class_mode='categorical', subset="validation")
model10 = models.Sequential()
model10.add(layers.Conv2D(filters= 100, kernel_size=(5, 5), activation='relu', input_shape=(64, 64, 3))) #(image_height, image_width, image_channels) (not including the batch dimension).
model10.add(layers.Conv2D(filters= 100, kernel_size=(5, 5), activation='relu'))
model10.add(layers.MaxPooling2D((4, 4)))
model10.add(layers.Flatten()) # Output_shape=(None, 3*3*64)
model10.add(layers.Dropout(0.1))
model10.add(layers.Dense(64, activation='relu'))
model10.add(layers.Dense(10, activation='relu'))

model10.compile(loss='categorical_crossentropy', optimizer=optimizers.adagrad(lr=0.001), metrics=['acc'])
history10 = model10.fit_generator(
        train_generator,
        steps_per_epoch=(11745 // 128),
        epochs=50,
        validation_data=val_generator,
        validation_steps=(2066 // 128) 
    )
acc = history10.history['acc']
val_acc = history10.history['val_acc']
loss = history10.history['loss']
val_loss = history10.history['val_loss']
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
model11 = models.Sequential()
model11.add(layers.Conv2D(filters= 100, kernel_size=(5, 5), activation='relu', input_shape=(64, 64, 3))) #(image_height, image_width, image_channels) (not including the batch dimension).
model11.add(layers.Conv2D(filters= 100, kernel_size=(5, 5), activation='relu'))
model11.add(layers.MaxPooling2D((4, 4)))
model11.add(layers.Flatten()) # Output_shape=(None, 3*3*64)
model11.add(layers.Dropout(0.3))
model11.add(layers.Dense(64, activation='relu'))
model11.add(layers.Dense(10, activation='relu'))

model11.compile(loss='categorical_crossentropy', optimizer=optimizers.adagrad(lr=0.001), metrics=['acc'])
history11 = model11.fit_generator(
        train_generator,
        steps_per_epoch=(11745 // 128),
        epochs=50,
        validation_data=val_generator,
        validation_steps=(2066 // 128) 
    )
acc = history11.history['acc']
val_acc = history11.history['val_acc']
loss = history11.history['loss']
val_loss = history11.history['val_loss']
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
model12 = models.Sequential()
model12.add(layers.Conv2D(filters= 100, kernel_size=(5, 5), activation='relu', input_shape=(64, 64, 3))) #(image_height, image_width, image_channels) (not including the batch dimension).
model12.add(layers.Conv2D(filters= 100, kernel_size=(5, 5), activation='relu'))
model12.add(layers.MaxPooling2D((4, 4)))
model12.add(layers.Flatten()) # Output_shape=(None, 3*3*64)
model12.add(layers.Dropout(0.5))
model12.add(layers.Dense(64, activation='relu'))
model12.add(layers.Dense(10, activation='relu'))

model12.compile(loss='categorical_crossentropy', optimizer=optimizers.adagrad(lr=0.001), metrics=['acc'])
history12 = model12.fit_generator(
        train_generator,
        steps_per_epoch=(11745 // 128),
        epochs=50,
        validation_data=val_generator,
        validation_steps=(2066 // 128) 
    )
acc = history12.history['acc']
val_acc = history12.history['val_acc']
loss = history12.history['loss']
val_loss = history12.history['val_loss']
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