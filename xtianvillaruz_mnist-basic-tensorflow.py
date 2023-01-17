# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import tensorflow as tf



tf.__version__
def load_data():

    with np.load("../input/mnist.npz") as f:

        x_train, y_train = f['x_train'], f['y_train']

        x_test, y_test = f['x_test'], f['y_test']

    return (x_train, y_train), (x_test, y_test)



(x_train, y_train), (x_test, y_test) = load_data()

print(x_train[0])
import matplotlib.pyplot as plt



plt.imshow(x_train[0], cmap = plt.cm.binary)

plt.show();
x_train = tf.keras.utils.normalize(x_train, axis=1)

x_test = tf.keras.utils.normalize(x_test, axis=1)

print(x_train[0])
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10)
val_loss, val_acc = model.evaluate(x_test, y_test)

print(val_loss, val_acc)



# Saving Model

# model.save('num_reader.model')



# Loading Model

# model = tf.keras.models.load_models('num_reader.model')
predictions = model.predict([x_test])



print(np.argmax(predictions[35]))

plt.imshow(x_test[35], cmap=plt.cm.binary)

plt.show();