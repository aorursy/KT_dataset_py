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

mnist = tf.keras.datasets.fashion_mnist



(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# Normalize

train_images = train_images / 255.0

test_images = test_images / 255.0
model = tf.keras.models.Sequential([

    tf.keras.layers.Flatten(), # 2d or 3d to 1d array(vector)

    tf.keras.layers.Dense(512, activation=tf.nn.relu), # i didn't give images shape bcz first layer output will be input for this layer # 512 Output

    tf.keras.layers.Dense(10, activation=tf.nn.softmax) # 10 output # softmax = multiclass

])



# relu: if greater than 0 then return that value or else return 0

# softmax: sum of every value than return probability (here it is 10 so it'll sum 10 values and then return probability of it, and which number is more then it'll  be our class)
model.compile(

    optimizer = tf.keras.optimizers.Adam(),

    loss = 'sparse_categorical_crossentropy',

    metrics = ['accuracy']

)
model.fit(

    train_images,

    train_labels,

    epochs = 5,

    batch_size = 128

)
model.evaluate(test_images, test_labels)