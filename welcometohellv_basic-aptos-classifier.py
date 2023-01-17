from __future__ import absolute_import

from __future__ import division

from __future__ import print_function
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import shutil

import csv

import tensorflow as tf

import keras_preprocessing

from keras_preprocessing import image

from keras_preprocessing.image import ImageDataGenerator
df_train=pd.read_csv(r"../input/train.csv",delimiter=',')

df_test=pd.read_csv(r"../input/test.csv",delimiter=',')

print(df_train.head())

print(df_test.head())

df_train['id_code']=df_train['id_code']+'.png'

#df_train['diagnosis']=df_train.astype({'diagnosis': 'category'})

df_train['diagnosis']=df_train['diagnosis'].astype(str)

#pd.get_dummies(df_train,prefix=['diagnosis'], drop_first=True)

df_test['id_code']=df_test['id_code']+'.png'
train_datagen = ImageDataGenerator(

    rescale = 1./255,

    rotation_range=30,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    fill_mode='nearest')



TRAINING_DIR='../input/train_images'



train_generator = train_datagen.flow_from_dataframe(

    dataframe=df_train,

    directory=TRAINING_DIR,

    x_col='id_code',

    y_col='diagnosis',

    batch_size=20,

    target_size=(1050,1050),

    class_mode='sparse'

)
from tensorflow.keras.optimizers import RMSprop
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(16 ,(5,5), activation='relu', input_shape=(1050,1050,3)),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(32,(2,2), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Conv2D(64,(5,5), activation='relu'),

    tf.keras.layers.MaxPooling2D(3,3),

    tf.keras.layers.Conv2D(64,(3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64,(5,5), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64,(3,3), activation='relu'),

    tf.keras.layers.Conv2D(64,(2,2), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64,(5,5), activation='relu'),

    tf.keras.layers.Dropout(0.05),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(1024, activation='relu'),

    tf.keras.layers.Dropout(0.05),

    tf.keras.layers.Dense(5, activation='softmax')

    

    

])



model.summary()

model.compile(loss='sparse_categorical_crossentropy',optimizer=RMSprop(lr=0.001),metrics=['accuracy'])
history=model.fit_generator(train_generator,steps_per_epoch=55 ,epochs=50,verbose=1)

model.save("first_part.h5")
import matplotlib.pyplot as plt

acc = history.history['acc']

#val_acc = history.history['val_acc']

loss = history.history['loss']

#val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'r', label='Training accuracy')

#plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

#plt.title('Training and validation accuracy')

plt.plot(epochs, loss, 'b', label='Training loss')

plt.title('Training')

plt.title('Loss')

plt.legend(loc=0)

plt.figure()





plt.show()