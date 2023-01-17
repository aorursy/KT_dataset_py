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
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images.shape
# train_images[0]

train_images[0].ndim
train_images[0].shape
train_labels[0]
train_labels[2]
from tensorflow.keras import models

from tensorflow.keras import layers



network = models.Sequential()

network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))

# MultiClassification Problem so "Softmax"

network.add(layers.Dense(10, activation='softmax'))

# probability will be check
network.compile(

    optimizer='rmsprop',

    loss='categorical_crossentropy',

    metrics=['accuracy']

)

# loss function
train_images = train_images.reshape((60000, 28 * 28)) # here converting 3d to 2d 

train_images = train_images.astype('float32') / 255 # all values will come under the range of 0 - 1



test_images = test_images.reshape((10000, 28 * 28))

test_images = test_images.astype('float32') / 255
from tensorflow.keras.utils import to_categorical



train_labels = to_categorical(train_labels)

test_labels = to_categorical(test_labels)
network.fit(

    train_images,

    train_labels,

    epochs=5, # no. of iteration

    batch_size=128 # chunk of input from total input

)
test_loss, test_acc = network.evaluate(test_images, test_labels)

print('Test Accuracy: ', test_acc)
from tensorflow.keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()



digit = train_images[4]



import matplotlib.pyplot as plt

plt.imshow(digit, cmap=plt.cm.binary)

plt.show()
train_labels[4]