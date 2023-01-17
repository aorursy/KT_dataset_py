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

data_datagen_e = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2,
                                    horizontal_flip=True, validation_split=0.15)

train_generator_e = data_datagen_e.flow_from_directory(
        "../input/simpsons_dataset/simpsons_dataset",
        classes=['homer_simpson', 'ned_flanders', 'moe_szyslak', 'lisa_simpson', 
                 'bart_simpson', 'marge_simpson', 'krusty_the_clown', 
                 'principal_skinner', 'charles_montgomery_burns', 'milhouse_van_houten', 'maggie_simpson'],
        target_size=(RESOLUTION, RESOLUTION),
        batch_size=BATCH_SIZE,
        class_mode='categorical', subset="training")

val_generator_e = data_datagen_e.flow_from_directory(
        "../input/simpsons_dataset/simpsons_dataset",
        classes=['homer_simpson', 'ned_flanders', 'moe_szyslak', 'lisa_simpson', 
                 'bart_simpson', 'marge_simpson', 'krusty_the_clown', 
                 'principal_skinner', 'charles_montgomery_burns', 'milhouse_van_houten', 'maggie_simpson'],
        target_size=(RESOLUTION, RESOLUTION),
        batch_size=BATCH_SIZE,
        class_mode='categorical', subset="validation")
model = models.Sequential()
model.add(layers.Conv2D(filters= 100, kernel_size=(5, 5), activation='relu', input_shape=(64, 64, 3))) #(image_height, image_width, image_channels) (not including the batch dimension).
model.add(layers.Conv2D(filters= 100, kernel_size=(5, 5), activation='relu'))
model.add(layers.MaxPooling2D((4, 4)))
model.add(layers.Flatten()) # Output_shape=(None, 3*3*64)
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(11, activation='relu'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer=optimizers.adagrad(lr=0.001), metrics=['acc'])

history = model.fit_generator(
        train_generator_e,
        steps_per_epoch=(11854 // 128),
        epochs=50,
        validation_data=val_generator_e,
        validation_steps=(2085 // 128) 
    )
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
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
