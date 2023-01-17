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
        os.path.join(dirname, filename)



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
print(f"Tensorflow:{tf.__version__}")
train_dir = "../input/intel-image-classification/seg_train/seg_train/"
test_dir = "../input/intel-image-classification/seg_test/seg_test/"
#train_dir = np.array(train_dir)
#test_dir = np.array(test_dir)
train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
                  train_dir,target_size = (150,150),batch_size=20,class_mode='categorical') 

test_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = test_datagen.flow_from_directory(
                  test_dir,target_size = (150,150),batch_size=20,class_mode='categorical')
layers = [
          tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation=tf.nn.relu, input_shape=(150,150,3)),
          tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation=tf.nn.relu),
          tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
          tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation=tf.nn.relu,),
          tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
          tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation=tf.nn.relu,),
          tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
          tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation=tf.nn.relu,),
          tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
          tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation=tf.nn.relu,),
          tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(units=512, activation=tf.nn.relu),
          tf.keras.layers.Dense(units=256, activation=tf.nn.relu),
          tf.keras.layers.Dense(units=6, activation=tf.nn.softmax),
]
model = tf.keras.Sequential(layers)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit_generator(train_generator,steps_per_epoch=35,epochs=35,
                              validation_data=validation_generator,
                              validation_steps=35,verbose=2)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

import matplotlib.pyplot as plt
plt.plot(epochs,acc)
plt.plot(epochs,val_acc)
plt.title("Training and validation Accuracy")
plt.figure()

plt.plot(epochs,loss)
plt.plot(epochs,val_loss)
plt.title("Training and validation Loss")
plt.figure()