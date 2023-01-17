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

import tensorflow.keras as k
import matplotlib.pyplot as plt

%matplotlib inline
raw_data = pd.read_csv("../input/train.csv")
data = raw_data.drop('label', axis=1)

data = data/255
label = raw_data['label']
image = data.iloc[20]

plt.imshow(image.values.reshape((28,28)), cmap='Blues')
input_x = tf.constant(data.values, dtype=tf.float32)
y = tf.one_hot(label, 10)
model = k.Sequential()
model.add(k.layers.Dense(512, input_shape=(784,), activation='relu'))

model.add(k.layers.Dropout(0.2))

model.add(k.layers.Dense(512, activation='relu'))

model.add(k.layers.Dropout(0.2))

model.add(k.layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(input_x, label, epochs=2, batch_size=None, steps_per_epoch=512)
y_test = pd.read_csv("../input/test.csv")
prediction = model.predict_classes(y_test.values)
results = pd.Series(prediction,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("results.csv",index=False)