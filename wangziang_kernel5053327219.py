# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import keras

from keras.applications.vgg16 import VGG16

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras import backend as K

from keras.optimizers import Adam, SGD, Adadelta, Adamax

from keras.metrics import categorical_crossentropy

from keras.models import load_model

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
train_path = '/kaggle/input/waste-pictures/train'

test_path = '/kaggle/input/waste-pictures/test'
train_datagen = ImageDataGenerator(rescale=1./255, 

                                   rotation_range=30, 

                                   zoom_range = 0.3, 

                                   width_shift_range=0.2,

                                   height_shift_range=0.2, 

                                   horizontal_flip = 'true')



test_datagen = ImageDataGenerator(rescale=1./255, 

                                  rotation_range=30, 

                                  zoom_range = 0.3, 

                                  width_shift_range=0.2,

                                  height_shift_range=0.2, 

                                  horizontal_flip = 'true')
train_generator = train_datagen.flow_from_directory(train_path, 

                                                    target_size=(224, 224), 

                                                    batch_size=128, 

                                                    class_mode='categorical')



test_generator = test_datagen.flow_from_directory(test_path, 

                                                    target_size=(224, 224), 

                                                    batch_size=128, 

                                                    class_mode='categorical')
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',

                 input_shape=(224, 224, 3)))

model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3, 3), padding='same'))

model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(128, (3, 3), padding='same'))

model.add(Activation('relu'))

model.add(Conv2D(128, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(256, (1, 1), padding='same'))

model.add(Activation('relu'))

model.add(Conv2D(256, (1, 1)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())

model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(128))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(34))

model.add(Activation('softmax'))


ad = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)

model.compile(loss='categorical_crossentropy',

              optimizer=ad,

              metrics=['accuracy'])
model.fit_generator(train_generator, 

                    shuffle=True,

                    epochs=50, 

                    validation_data=test_generator)
model.save("new_model.h5")