# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_gen=tf.keras.preprocessing.image.ImageDataGenerator(

            rescale=1./255,

            horizontal_flip=True)

test_gen=tf.keras.preprocessing.image.ImageDataGenerator(

            rescale=1./255,

            horizontal_flip=True)

train_dir='../input/chest-xray-pneumonia/chest_xray/chest_xray/train'

train_generator=train_gen.flow_from_directory(

                train_dir,

                class_mode='binary')
validation_dir='../input/chest-xray-pneumonia/chest_xray/chest_xray/val'

validation_generator=train_gen.flow_from_directory(

                validation_dir,

                class_mode='binary')
model=tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(8,(4,4),activation='relu'),

    tf.keras.layers.MaxPooling2D(3,3),

    tf.keras.layers.Conv2D(8,(4,4),activation='relu'),

    tf.keras.layers.MaxPooling2D(3,3),

    tf.keras.layers.Conv2D(8,(4,4),activation='relu'),

    tf.keras.layers.MaxPooling2D(3,3),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(1024,activation='relu'),

    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(256,activation='relu'),

    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(128,activation='relu'),

    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(1,activation='sigmoid'),

    

])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
train=model.fit_generator(

    train_generator,

    epochs=5,

    validation_data=validation_generator                          

)
accuracy=train.history['acc']

val_acc=train.history['val_acc']

loss=train.history['loss']

val_loss=train.history['val_loss']

epochs=list(range(5))
import matplotlib.pyplot as plt

plt.plot(epochs,accuracy)

plt.xlabel('EPOCHS')

plt.ylabel('ACCURACY')

plt.plot(epochs,val_acc)

plt.legend(['TRAINING ACC','VAL ACC'])

plt.show()
import matplotlib.pyplot as plt

plt.plot(epochs,loss)

plt.xlabel('EPOCHS')

plt.ylabel('LOSS')

plt.plot(epochs,val_loss)

plt.legend(['TRAINING LOSS','VAL LOSS'])

plt.show()
print("Accuracy : ",accuracy[-1]*100)