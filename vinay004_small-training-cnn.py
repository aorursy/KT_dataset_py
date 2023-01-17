import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from os import listdir, makedirs
from os.path import join, exists, expanduser
import cv2
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dense, GlobalAveragePooling2D,Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense,BatchNormalization
from keras import backend as K
from keras.datasets import cifar10
import tensorflow as tf
from scipy import io
from keras.utils import np_utils
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
(train_images,train_labels),(test_images,test_labels) = cifar10.load_data()
print(np.shape(train_images),np.shape(train_labels))
print(np.shape(test_images),np.shape(test_labels))
batch_size = 64
num_classes = 10
train_labels = np_utils.to_categorical(train_labels,num_classes)
test_labels = np_utils.to_categorical(test_labels,num_classes)
for i in range(len(train_images)):
    train_images[i] = cv2.normalize(train_images[i], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
for i in range(len(test_images)):    
    test_images[i]  = cv2.normalize(test_images[i], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
val_images = train_images[-10000:]
val_labels = train_labels[-10000:]
train_images = train_images[:-10000]
train_labels = train_labels[:-10000]
model = Sequential()
model.add(Conv2D(32, (3, 3),input_shape=train_images.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides = 2))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides = 2))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.summary()
opt = optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
datagen = ImageDataGenerator(
        zca_whitening=True,
        zca_epsilon=1e-06, 
        rotation_range=45, 
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        data_format=None,
        validation_split=0.2)

datagen.fit(train_images)
model.fit_generator(datagen.flow(train_images, train_labels,batch_size=batch_size),
                    epochs=15,validation_data=(val_images, val_labels),workers=4)
model.evaluate(test_images, test_labels, verbose=1)
