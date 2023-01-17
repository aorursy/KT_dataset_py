os.listdir('../input/melanoma')
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
import keras 
from keras import models
from keras.applications import ResNet50
from keras.layers import Dense, Dropout, Flatten
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import os
import numpy as np

import cv2
import matplotlib.pyplot as  plt
from keras import models
from tensorflow.keras.models import load_model,model_from_json
conv_base = ResNet50(weights='imagenet',
include_top=False,
input_shape=(224, 224, 3))

print(conv_base.summary())
model = models.Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

print(model.summary())
# Make last block of the conv_base trainable:

for layer in conv_base.layers[:165]:
   layer.trainable = False
for layer in conv_base.layers[165:]:
   layer.trainable = True

print('Last block of the conv_base is now trainable')
for i, layer in enumerate(conv_base.layers):
   print(i, layer.name, layer.trainable)
model.compile(optimizer=Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("model compiled")
print(model.summary())
# Prep the Train Valid and Test directories for the generator

train_dir = '../input/melanoma/dermmel/DermMel/train_sep'
validation_dir = '../input/melanoma/dermmel/DermMel/valid'
batch_size = 20
target_size=(224, 224)

#train_datagen = ImageDataGenerator(rescale=1./255)
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,target_size=target_size,batch_size=batch_size)
validation_generator = val_datagen.flow_from_directory(
    validation_dir,target_size=target_size,batch_size=batch_size)
#training the model
history = model.fit_generator(train_generator,
                              epochs=20,
                              steps_per_epoch = 10682 ,
                              validation_data = validation_generator,
                              validation_steps = 3562 )