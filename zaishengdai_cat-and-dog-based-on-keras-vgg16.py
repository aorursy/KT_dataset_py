# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/cat-and-dog"))



# Any results you write to the current directory are saved as output.
# leverage existing VGG16 to classify cats and dogs

from keras import backend as K

from keras import applications

from keras import Sequential

from keras import Model

from keras.layers import Dropout, Flatten, Dense

from keras.preprocessing.image import ImageDataGenerator

from keras import optimizers

from keras import backend as K

import os



img_width, img_height = 150, 150

input_shape=(img_width, img_height, 3)

from keras.models import load_model

model = load_model('../input/vgg16h5/vgg16.h5')

#model = applications.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

for layer in model.layers[:15]:

    layer.trainable = False

top_model = Flatten(input_shape=model.output_shape[1:])(model.output)

top_model = Dense(256, activation='relu')(top_model)

top_model = Dropout(0.5)(top_model)

top_model = Dense(1, activation='sigmoid')(top_model)

model = Model(inputs=model.input, outputs=top_model)

model.compile(loss='binary_crossentropy',

              optimizer=optimizers.SGD(lr=1e-2, momentum=0.9),

              metrics=['accuracy'])
train_data_dir = "../input/cat-and-dog/training_set"

validation_data_dir = "../input/cat-and-dog/test_set"

epochs = 10

batch_size = 16

# prepare data augmentation configuration

train_datagen = ImageDataGenerator(

    rescale=1/255,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True)



test_datagen = ImageDataGenerator(rescale=1/255)



train_generator = train_datagen.flow_from_directory(

    train_data_dir,

    target_size=(img_height, img_width),

    batch_size=batch_size,

    class_mode='binary')



validation_generator = test_datagen.flow_from_directory(

    validation_data_dir,

    target_size=(img_height, img_width),

    batch_size=batch_size,

    class_mode='binary')



# fine-tune the model

model.fit_generator(

    train_generator,

    steps_per_epoch=16,

    epochs=epochs,

    validation_data=validation_generator,

    validation_steps=200)