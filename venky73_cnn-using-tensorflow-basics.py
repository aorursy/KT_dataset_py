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

from tensorflow import keras
train = pd.read_csv("../input/fashion-mnist_train.csv")

test = pd.read_csv("../input/fashion-mnist_test.csv")
train_y = list(train.label)

train_x = train.drop(columns = ["label"])

train_x = np.array(train_x)

train_x = train_x/255

train_x = train_x.reshape(60000, 28, 28, 1)

len(train_y)


train_x.shape
#step - 1

model = tf.keras.models.Sequential([

  tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),

  tf.keras.layers.MaxPooling2D(2, 2),

  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

  tf.keras.layers.MaxPooling2D(2,2),

  tf.keras.layers.Flatten(),

  tf.keras.layers.Dense(128, activation='relu'),

  tf.keras.layers.Dense(10, activation='softmax')

])



#step - 2

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

#step - 3

model.fit(train_x, train_y, epochs=5)



out = list(model.predict(train_x[0].reshape(1,28,28,1))[0])

out.index(max(out))
train[:1]["label"]