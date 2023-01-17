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
# Loading the the important packages 

import tensorflow as tf

import PIL

from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, Dense, Flatten, Dropout, Input

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.losses import CategoricalCrossentropy

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Model

from tensorflow.keras.applications import InceptionV3
def data_generation(training_path, validation_path, target_size):

    data_gen = ImageDataGenerator(rescale=1.0 / 255.)

    

    training_datagen = data_gen.flow_from_directory(training_path,

                                                    target_size= target_size,

                                                    batch_size = 50,

                                                    class_mode='categorical'

                                                    )

    

    validation_datagen = data_gen.flow_from_directory(validation_path,

                                                    target_size= target_size,

                                                    class_mode='categorical'

                                                    )

    return training_datagen, validation_datagen

    
# Image Preview:

image_path = '/kaggle/input/image-classification/test/test/classify/8.JPG'

img = PIL.Image.open(image_path)

img_numpy = np.array(img)

print("The shape of the image is: ",img_numpy.shape)

img
# Global Variables

training_path = '/kaggle/input/image-classification/images/images/'

validation_path = '/kaggle/input/image-classification/validation/validation/'

target_size = (220,220)
training, validation = data_generation(training_path,validation_path, target_size)
inception_model = InceptionV3(include_top=False, input_shape=(220,220,3))
inception_model.summary()
for layer in inception_model.layers:

    layer.trainable = False

last_layer = inception_model.get_layer('mixed10').output
def full_model():

    X = Flatten()(last_layer)

    X = Dense(1024,activation = 'relu')(X)

    X = Dropout(0.3)(X)

    X = Dense(512,activation = 'relu')(X)

    X = Dropout(0.3)(X)

    X = Dense(256,activation='relu')(X)

    X = Dropout(0.2)(X)

    X = Dense(4,activation = 'softmax')(X)

    model = Model(inception_model.input, X)

    return model
model = full_model()

model.summary()
model.compile(optimizer=Adam(0.0001),loss=CategoricalCrossentropy(),metrics=['accuracy'])
history = model.fit_generator(training,validation_data=validation, epochs = 2)