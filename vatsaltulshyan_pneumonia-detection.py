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
train_dir = '/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train/'
test_dir  = '/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/test/'
val_dir   = '/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/val/'
import tensorflow as tf
train_data = tf.io.gfile.glob(train_dir+'/*/*')
val_data = tf.io.gfile.glob(val_dir + '/*/*')
files = train_data
files.extend(val_data)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen2=ImageDataGenerator(rescale=1.0/255,rotation_range=30,width_shift_range=0.2,height_shift_range=0.2,zoom_range=0.2)
val_datagen2=ImageDataGenerator(rescale=1.0/255)
test_datagen2=ImageDataGenerator(rescale=1.0/255)
train_generator2=train_datagen2.flow_from_directory(train_dir,target_size=(180,180),batch_size=128,class_mode='binary')
val_generator2=val_datagen2.flow_from_directory(val_dir,target_size=(180,180),batch_size=128,class_mode='binary')
test_generator2=test_datagen2.flow_from_directory(test_dir,target_size=(180,180),batch_size=128,class_mode='binary')
from tensorflow.keras.applications.xception import Xception
model = Xception(weights= None, include_top=False, input_shape= (180,180,3))
for layers in model.layers:
    layers.trainable = False
model.summary()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
mob_model = Sequential()
mob_model.add(tf.keras.applications.MobileNetV2(include_top=False, pooling = 'avg', weights='imagenet',input_shape=(224, 224, 3), classes=2))
mob_model.add(Dense(32, activation='relu'))
mob_model.add(Dense(1, activation='sigmoid'))
mob_model.layers[0].trainable = False
mob_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
mob_model.fit(train_generator2,validation_data=test_generator2,epochs=10,steps_per_epoch=5,verbose=2)
mob_model.save("saved-model")


cnn_model = mob_model.fit_generator(training_set,
                         steps_per_epoch = 163,
                         epochs = 1,
                         validation_data = validation_generator,
                         validation_steps = 624)
test_accu = cnn.evaluate_generator(test_set,steps=624)

