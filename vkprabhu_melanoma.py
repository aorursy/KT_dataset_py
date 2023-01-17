# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    #print(dirname)

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Importing the necessary libraries for Deep Learning

from keras.layers import Input, Lambda, Dense, Flatten, Dropout

from keras.optimizers import Adam

from keras import regularizers

from keras.models import Model

from keras.applications import DenseNet121, InceptionResNetV2, ResNet152V2, ResNet50, DenseNet121

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

import matplotlib.pyplot as plt

from keras.backend import image_data_format
if image_data_format() == 'channels_last':

    input_shape = (224, 224, 3)

else:

    input_shape = (3, 224, 224)
print(input_shape)
conv_base = DenseNet121(weights='imagenet',

include_top=False,

input_shape=input_shape)



print(conv_base.summary())
print(len(conv_base.layers))
model = Sequential()

model.add(conv_base)

model.add(Flatten())

model.add(Dropout(0.5))

model.add(Dense(64, activation='relu',kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(0.001)))

model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))



print(model.summary())
# Make last block of the conv_base trainable:



for layer in conv_base.layers[:300]:

   layer.trainable = False

for layer in conv_base.layers[300:]:

   layer.trainable = True



print('Last block of the conv_base is now trainable')
model.compile(optimizer=Adam(lr=1e-5),

              loss='binary_crossentropy',

              metrics=['accuracy'])



print("model compiled")

print(model.summary())
# data agumentation for training data

train_datagen = ImageDataGenerator(rescale=1./255,

                                   rotation_range=40,

                                   width_shift_range=0.2,

                                   height_shift_range=0.2,

                                   shear_range=0.2,

                                   zoom_range=0.2,

                                   horizontal_flip=True,

                                   vertical_flip=True,

                                   fill_mode='nearest')
# normalizing validation data

validation_datagen = ImageDataGenerator(rescale = 1./255)
# normalize test data for making predictions

test_datagen = ImageDataGenerator(rescale = 1./255)
# Traing data geneartor

batch_size = 20

training_set = train_datagen.flow_from_directory('/kaggle/input/melanoma/dermmel/DermMel/train_sep',

                                                 target_size = (224, 224),

                                                 batch_size = batch_size,

                                                 class_mode = 'binary')



validation_set = validation_datagen.flow_from_directory('/kaggle/input/melanoma/dermmel/DermMel/valid/',

                                            target_size = (224, 224),

                                            batch_size = batch_size,

                                            class_mode = 'binary')

test_set = test_datagen.flow_from_directory('/kaggle/input/melanoma/dermmel/DermMel/test/',

                                            target_size = (224, 224),

                                            batch_size = batch_size,

                                            class_mode = 'binary'

                                            )
# view the structure of the model

model.summary()
# fit the model

model.fit_generator(training_set,

                              epochs = 10,

                              steps_per_epoch = 10682 // batch_size,

                              validation_data = validation_set,

                              validation_steps = 3562 // batch_size)
loss,accuracy = model.evaluate_generator(test_set,

                                      steps = len(test_set) // batch_size,

                                      workers = -1,

                                      use_multiprocessing = True,

                                      verbose = 1)
print('The accuracy of the model: {}'.format(accuracy))
# Try using other transfer learning techniques to improve the accuracy of the models.