import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import random

import os

from keras.preprocessing.image import ImageDataGenerator, load_img

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

import cv2
PATH = "./destination_path/train"
from zipfile import ZipFile

with ZipFile('../input/dogs-vs-cats/train.zip', 'r') as zf:

    zf.extractall('destination_path/')
IMG_SIZE = 128

def create_data():

    x = []

    y = []

    for file_name in os.listdir(PATH):

        img_array = cv2.imread(os.path.join(PATH, file_name))

        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

        x.append(new_array)

        if file_name.split('.')[0] == 'cat':

            y.append(0)

        else:

            y.append(1)

    return x, y
X,y = create_data()

X = np.array(X)

y = np.array(y)
plt.imshow(X[30],interpolation='nearest', cmap='Greys')
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
IMAGE_WIDTH=128

IMAGE_HEIGHT=128

IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)

IMAGE_CHANNELS=3
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(128, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary()
earlystop = EarlyStopping(patience=10)
X_train, X_test, y_train, y_test = train_test_split(X,y)
X_train = X_train / 255.0

Y_train = to_categorical(y_train, num_classes = 2)
history =  model.fit(x=X_train, y=Y_train, epochs=10, validation_split=0.2,callbacks=[earlystop])
val_acc = history.history['val_accuracy']



loss = history.history['loss']

val_loss = history.history['val_loss']



plt.figure(figsize=(14, 8))

plt.subplot(1, 2, 1)

plt.plot(acc, label='Training Accuracy')

plt.plot(val_acc, label='Validation Accuracy')

plt.legend(loc = 'upper right')

plt.ylabel('Accuracy')

# plt.ylim([0.5, 1.0])

plt.title('Training and Validation Accuracy')



plt.subplot(1, 2, 2)

plt.plot(loss, label='Training Loss')

plt.plot(val_loss, label='Validation Loss')

plt.legend(loc='upper right')

plt.ylabel('Cross Entropy')

# plt.ylim([0.5,1.0])

plt.title('Training and Validation Loss')