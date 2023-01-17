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
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
mnist = tf.keras.datasets.mnist         #Loading Data from Mnist Dataset from Keras
(x_train,y_train),(x_test,y_test)=mnist.load_data() #Splitting into training and testing Dataset

x_train=tf.keras.utils.normalize(x_train,axis=1)      #Normalizing the Dataset
x_test=tf.keras.utils.normalize(x_test,axis=1)
x_test
model = Sequential()                                 
model.add(CuDNNLSTM(128,input_shape=(28,28),return_sequences=True))            #Adding an LSTM (Long Short Term Memory) Layer (We dont keep the activation function as it takes default tanh function)
model.add(Dropout(0.3))            #Adding Dropout to keep the model generalized
model.add(CuDNNLSTM(128))
model.add(Dropout(0.2))

model.add(Dense(10,activation='softmax'))             #Last layer output of 10 neurons with softmax activation

opt = tf.keras.optimizers.Adam(lr=1e-3,decay=1e-5)  #herer decay is rate of decrease in lr
model.compile(loss='sparse_categorical_crossentropy',                     #Here sparse categorical loss is used due to 10 outputs
             optimizer=opt,
             metrics=['accuracy'])
model.fit(x_train,y_train,epochs=10,validation_data=(x_test,y_test))         #increasing the epochs increases the accuracy

#THIS MODEL CAN BE USED AND JUST BY TRANSFORMING THE TEST DATA A GOOD ACCURACY CAN BE ACHIEVED
