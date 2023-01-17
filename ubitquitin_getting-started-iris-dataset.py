# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import seaborn as sns
import os
import tensorflow as tf
from tensorflow import keras
from keras import Model
# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/iris.csv")
df.head()
df.info()
df.describe()
df['Species'].value_counts()
sns.pairplot(df,hue ='Species')
#Splitting data into training and testing categories as well as input/output
training_data = df[0:104]
training_results = training_data['Species']
training_data = training_data.drop(columns='Species')

testing_data = df[105::]
testing_results = testing_data['Species']
testing_data = testing_data.drop(columns='Species')
#normalize data
training_data = (training_data - training_data.mean())/training_data.std()
training_data.describe()

sns.pairplot(training_data)
#initialize a sequential model object
model = keras.Sequential()

#add input layer
#Add a Dense layer with the input shape of the training data.Use
#a reLu function for nonlinear activation.
model.add(keras.layers.Dense(64, activation=tf.nn.relu,
                       input_shape=(training_data.shape[1],)))

#add hidden layer
#Add a Dense layer again using a reLu function.
model.add(keras.layers.Dense(64, activation=tf.nn.relu))

model.add(keras.layers.Dense(1))

model.summary()
#RMSProp used to speed up gradient descent
optimizer = tf.train.RMSPropOptimizer(0.001)

#mean squared error loss(error) function to be minimized for the NN.
model.compile(loss='mse',
        optimizer=optimizer,
        metrics=['accuracy'])

history = model.fit(training_data, training_results, epochs=500,
                    validation_split=0.3, verbose=0)

#TODO: Convert training_reults into floating points? Review optimizers and 
#loss functions.
