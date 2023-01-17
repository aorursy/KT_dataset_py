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
from keras.preprocessing.image import ImageDataGenerator 

from keras.models import Sequential 

from keras.layers import Conv2D, MaxPooling2D 

from keras.layers import Activation, Dropout, Flatten, Dense 

from keras import backend as K 

import numpy as np 

from keras.preprocessing import image
img_width, img_height = 150,150



train_data_dir = "../input/train"

test_data_dir = "../input/test"



num_train_samples = 1000

num_val_samples = 100

epochs = 50

batch_size = 20



#if K.common.image_data_format() =='channels_first':

 #   input_shape = (3, img_height, img_height)

#else:

input_shape = (img_width, img_height,3)
train_datagen = ImageDataGenerator(

        rescale=1./255,

        zoom_range=0.2,

        horizontal_flip=True,

        shear_range=0.2)



test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

    train_data_dir,

    target_size = (img_width, img_height),

    batch_size = batch_size,

    class_mode = 'binary')



validation_generator = test_datagen.flow_from_directory(

    test_data_dir,

    target_size = (img_width, img_height),

    batch_size = batch_size,

    class_mode = 'binary')
model = Sequential()

model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape))

model.add(MaxPooling2D(2,2))



model.summary()

print('\n')



model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))

model.add(MaxPooling2D(2,2))



model.summary()

print('\n')



model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))

model.add(MaxPooling2D(2,2))



model.summary()

print('\n')



model.add(Flatten())

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(1, activation='relu'))



model.summary()
model.compile(loss='binary_crossentropy',

             optimizer='rmsprop',

             metrics=['accuracy'])



model.fit_generator(train_generator, 

                   steps_per_epoch=num_train_samples // batch_size,

                   epochs=epochs,

                   validation_data=validation_generator,

                   validation_steps=num_val_samples // batch_size)