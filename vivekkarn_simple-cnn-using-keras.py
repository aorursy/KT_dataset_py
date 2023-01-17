# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
from glob import glob
import random
import cv2
import matplotlib.pylab as plt
import keras
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Lambda, MaxPool2D, BatchNormalization, Input
from keras.models import Sequential
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from pathlib import Path
from keras.optimizers import Adam,RMSprop,SGD
# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/10-monkey-species/monkey_labels.txt")
df
height=150
width=150
channels=3
batch_size=32
seed=1337

train_dir = Path('../input/10-monkey-species/training/training/')
test_dir = Path('../input/10-monkey-species/validation/validation/')

# Training generator
train_datagen = ImageDataGenerator(rotation_range = 30
                                   ,rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(train_dir, 
                                                    target_size=(height,width),
                                                    batch_size=batch_size,
                                                    seed=seed,
                                                    class_mode='categorical')

# Test generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(test_dir, 
                                                  target_size=(height,width), 
                                                  batch_size=batch_size,
                                                  seed=seed,
                                                  class_mode='categorical')
weights = Path('../input/VGG-16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
from keras.applications.vgg16 import VGG16
conv_base = VGG16(include_top= False, weights = '../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                 input_shape=(150,150,3))
conv_base.summary()
model = Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256,activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(Adam(lr=0.0001),loss="categorical_crossentropy", metrics=["accuracy"])
#model.fit(train_x, train_y, batch_size=32, validation_data = (test_x, test_y), epochs=10)
history = model.fit_generator(train_generator,
                    steps_per_epoch = 1097//batch_size,
                    validation_data = test_generator,
                    validation_steps = 4,
                    epochs = 10,
                    verbose  = 2)
model.summary()
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.title('Training and validation accuracy')
plt.plot(epochs, acc, 'red', label='Training acc')
plt.plot(epochs, val_acc, 'blue', label='Validation acc')
plt.legend()

plt.figure()
plt.title('Training and validation loss')
plt.plot(epochs, loss, 'red', label='Training loss')
plt.plot(epochs, val_loss, 'blue', label='Validation loss')

plt.legend()

plt.show()

