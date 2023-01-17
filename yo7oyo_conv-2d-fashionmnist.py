# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install tensorflow-gpu==2.0.0
import tensorflow as tf

tf.__version__
tf.test.is_gpu_available()
from tensorflow import keras

import matplotlib.pyplot as plt

import numpy as np
(train_img, train_labels), (test_img, test_labels) = keras.datasets.fashion_mnist.load_data()
train_img = np.expand_dims(train_img, axis=-1)

train_img.shape
test_img = np.expand_dims(test_img, axis=-1)

test_img.shape
model = keras.Sequential()

model.add(keras.layers.Conv2D(32, (3, 3), input_shape=train_img.shape[1:], activation='relu'))

model.add(keras.layers.MaxPool2D(padding='same'))

model.add(keras.layers.Conv2D(64, (3, 3), activation='relu')) 

model.add(keras.layers.GlobalAveragePooling2D())  

model.add(keras.layers.Dense(10, activation='softmax'))
model.summary()
model.compile(optimizer='adam',

             loss='sparse_categorical_crossentropy',

             metrics=['acc'])
history = model.fit(train_img, train_labels, validation_data=(test_img, test_labels))