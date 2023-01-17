import os

import cv2

import numpy as np

import matplotlib.pyplot as plt

import random

import tensorflow as tf

import keras

from keras.models import Sequential, Model

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D

from keras import applications

from keras import backend as K

from keras.optimizers import SGD

from keras import regularizers

from keras.layers.normalization import BatchNormalization

from keras import optimizers

import os

from os import listdir

from PIL import Image

from keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img

import numpy as np
train_data = []

train_labels = []

test_data = []

test_labels = []

im_width = 128

im_height = 128

num_classes = 120
def preProcessTrainData(path):

    i = 0

    progress = 0

    folders = [f for f in os.listdir(path)]

    for folder in folders:

        image_files = [f for f in os.listdir(path+folder)]

        for file_name in image_files:

            image_file = str(path + folder+'/' +file_name)

        

            img = cv2.imread(image_file,cv2.IMREAD_GRAYSCALE)

            new_img = cv2.resize(img,(im_width,im_height))

            train_data.append(new_img)

            progress = progress+1

        

            train_labels.append(i)

        

            if progress%1000==0:

                print('Progress '+str(progress)+' Image done')

        i = i + 1

print(os.listdir("../input"))
preProcessTrainData("../input/images/Images/")
train_data = np.array(train_data)

print(train_data.shape)
train_labels = np.array(train_labels)

train_labels.shape
train_data = train_data.reshape((train_data.shape)[0],(train_data.shape)[1],(train_data.shape)[2],1)

train_data.astype('float32')

train_data = train_data/255.0
train_labels.astype('uint8')

#test_labels.astype('uint8')

train_labels = keras.utils.to_categorical(train_labels, num_classes)

#test_labels = keras.utils.to_categorical(test_labels, num_classes)
def shuffle(a, b):

    rng_state = np.random.get_state()

    np.random.shuffle(a)

    np.random.set_state(rng_state)

    np.random.shuffle(b)



for i in range(10):

    shuffle(train_data,train_labels)
model = Sequential()

model.add(Conv2D(kernel_size=(3,3),filters=32,input_shape=(128, 128,

1),activation="relu",padding="valid"))



model.add(Conv2D(kernel_size=(3,3),filters=32,activation="relu",padding="same"))

model.add(Dropout(0.15))



model.add(Conv2D(kernel_size=(3,3),filters=24))

model.add(Conv2D(kernel_size=(3,3),filters=64,activation="relu",padding="same"))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(kernel_size=(3,3),filters=24))

model.add(Dropout(0.25))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(kernel_size=(5,5),filters=32,activation="relu",padding="same"))



model.add(MaxPooling2D(pool_size=(3,3)))





model.add(Flatten())

model.add(Dense(100,activation="relu",kernel_regularizer=keras.regularizers.l2(0.01)))

model.add(Dropout(0.4))

model.add(Dense(num_classes,activation="softmax"))



model.compile(loss=keras.losses.binary_crossentropy,

              optimizer=keras.optimizers.Adadelta(lr=0.01),

              metrics=['accuracy'])

model.summary()
history = model.fit(train_data, train_labels,

          batch_size=100,

          epochs=10,

          verbose=1, shuffle = True,validation_split=0.15)
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['train', 'test'], loc='upper right')

plt.show()