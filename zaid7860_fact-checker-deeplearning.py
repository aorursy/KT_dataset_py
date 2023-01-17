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


import numpy as np

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Dropout, Flatten, Dense

from keras import applications

import matplotlib.pyplot as plt
img_width, img_height = 256, 256

train_data_dir = '../input/factchecked-images-shared-during-electionsind/india'
batch_size=32





train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(

     rescale=1./255,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True,

    validation_split=0.2,



)

train_data_gen = train_datagen.flow_from_directory(

                                                            train_data_dir,

    target_size=(img_height, img_width),

    batch_size=batch_size,

    class_mode='categorical',

    subset='training') #



val_data_gen = train_datagen.flow_from_directory(train_data_dir, # same directory as training data

    target_size=(img_height, img_width),

    batch_size=batch_size,

    class_mode='categorical',

    subset='validation')
sample_training_images, _ = next(train_data_gen)

# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.

def plotImages(images_arr):

    fig, axes = plt.subplots(1, 5, figsize=(20,20))

    axes = axes.flatten()

    for img, ax in zip( images_arr, axes):

        ax.imshow(img)

        ax.axis('off')

    plt.tight_layout()

    plt.show()
plotImages(sample_training_images[:5])

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
my_model = Sequential()

my_model.add(Conv2D(64, kernel_size=4, strides=1, activation='relu', input_shape=(img_width, img_height,3)))

my_model.add(Conv2D(64, kernel_size=4, strides=2, activation='relu'))

my_model.add(Dropout(0.5))

my_model.add(Conv2D(128, kernel_size=4, strides=1, activation='relu'))

my_model.add(Conv2D(128, kernel_size=4, strides=2, activation='relu'))

my_model.add(Dropout(0.5))

my_model.add(Conv2D(256, kernel_size=4, strides=1, activation='relu'))

my_model.add(Conv2D(256, kernel_size=4, strides=2, activation='relu'))

my_model.add(Flatten())

my_model.add(Dropout(0.5))

my_model.add(Dense(512, activation='relu'))

my_model.add(Dense(3, activation='softmax'))



my_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])

my_model.summary()

from keras.callbacks import ModelCheckpoint, EarlyStopping

checkpoint = ModelCheckpoint("news.h5",

                             monitor="val_loss",

                             mode="min",

                             save_best_only = True,

                             verbose=1)



earlystop = EarlyStopping(monitor = 'val_loss', 

                          min_delta = 0, 

                          patience = 3,

                          verbose = 1,

                          restore_best_weights = True)



# we put our call backs into a callback list

callbacks = [earlystop, checkpoint]
history = my_model.fit_generator(

    train_data_gen,

    steps_per_epoch=16, #batch_size

    epochs=50,

    callbacks = callbacks,

    validation_data=val_data_gen,

    validation_steps=8 

)
import matplotlib.pyplot as plt



def plot_graphs(history, metric):

  plt.plot(history.history[metric])

  plt.plot(history.history['val_'+metric], '')

  plt.xlabel("Epochs")

  plt.ylabel(metric)

  plt.legend([metric, 'val_'+metric])

  plt.show()
plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')




