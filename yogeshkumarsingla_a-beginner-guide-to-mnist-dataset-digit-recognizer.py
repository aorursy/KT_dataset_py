# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

np.random.seed(1212)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import tensorflow as tf

import keras



# Reading train and test data

train_data = pd.read_csv('../input/digit-recognizer/train.csv')

test_data = pd.read_csv('../input/digit-recognizer/test.csv')



# Class names

val_data = train_data.iloc[:5000,:]

train_data = train_data.iloc[5000:,:]



# Fetching the labels

train_labels = train_data.label

val_labels = val_data.label



# Reshaping training data

train_images = train_data.iloc[:,1:].values.reshape(37000, 28, 28)

# Reshaping validation data

val_images = val_data.iloc[:,1:].values.reshape(5000, 28, 28)



# Scaling data in the range of 0-1

train_images = train_images/255.0

val_images = val_images/255.0







# Defining multi-layer perceptron model with 3 hidden layer having 10 neurons each and with non-linearity

model = keras.Sequential([

    keras.layers.Flatten(input_shape=(28, 28)), # Perform conversion of higher dimensional data (here, 2-D) to 1-D data.

    keras.layers.Dense(10, activation=tf.nn.leaky_relu), # Hidden layer with 10 neurons and ReLU activation function

    keras.layers.Dense(10, activation=tf.nn.leaky_relu), # Hidden layer with 10 neurons and ReLU activation function

    keras.layers.Dense(10, activation=tf.nn.leaky_relu), # Hidden layer with 10 neurons and ReLU activation function

    keras.layers.Dense(10, activation=tf.nn.leaky_relu), # Hidden layer with 10 neurons and ReLU activation function

    keras.layers.Dense(10, activation=tf.nn.softmax) # Output layer with softmax activation function 

])

# Defining parameters like optimizer, loss function and evaluating metric

model.compile(loss='sparse_categorical_crossentropy',

              optimizer=keras.optimizers.Adam(),

              metrics=['accuracy'])

model4 = model.fit(train_images, train_labels, epochs=20, validation_data=(val_images, val_labels))





test_images = test_data.iloc[:,:].values.reshape(28000, 28, 28)

test_images = test_images/255.0



output = model.predict(test_images)

results = np.argmax(output,axis = 1)

results = pd.Series(results,name="Label")



submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("mnist_submission.csv",index=False)
