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
def generator():
    data_generator=tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=45,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    
    return data_generator
train_generator=generator()
test_generator=generator()
def data_flow(path,generator):
    data_flow=train_generator.flow_from_directory(path,batch_size=100,class_mode='binary')
    return data_flow;
train_dir='../input/hot-dog/hotdog/train'
train_data_generator=data_flow(train_dir,train_generator)
test_dir='../input/hot-dog/hotdog/test'
test_data_generator=data_flow(test_dir,train_generator)
model=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(3,3),
    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(3,3),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(3,3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256,activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(16,activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(1,activation='sigmoid')
    
])
model.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.RMSprop(lr=0.001),metrics=['acc'])
model_training=model.fit_generator(train_data_generator,
                                   epochs=15,
                                   validation_data=test_data_generator,
                                   validation_steps=8)
accuracy=model_training.history['acc']
validation_acc=model_training.history['val_acc']
loss=model_training.history['loss']
validation_loss=model_training.history['val_loss']
import matplotlib.pyplot as plt
plt.plot(list(range(15)),accuracy,label="Accuracy")
plt.plot(list(range(15)),validation_acc,label="Validation Accracy")
plt.xlabel("EPOCHS")
plt.ylabel("Validation Accuracy")
plt.legend()
plt.show()
plt.plot(list(range(15)),loss,label="Loss")
plt.plot(list(range(15)),validation_loss,label="Validation Loss")
plt.xlabel("EPOCHS")
plt.ylabel("Validation Loss")
plt.legend()
plt.show()
plt.bar(list(range(15)),accuracy)
plt.xlabel("EPOCHS")
plt.ylabel("Accuracy")
plt.show()
model.save('model')
import shutil

zip_name = 'trained_model'
directory_name = 'model'

shutil.make_archive(zip_name, 'zip', directory_name)
