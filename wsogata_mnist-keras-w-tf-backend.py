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
#Import packages

import keras
from keras import regularizers
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout 
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
import numpy as np
import pandas as pd

np.random.seed(44)
# Load Data
#(X_train, y_train),(X_test, y_test) = mnist.load_data()

#somehow the above throws an exception: "URL fetch failure on https://s3.amazonaws.com/img-datasets/mnist.npz: None -- [Errno -2] Name or service not known"

# Assign np data manually after uploading to Kaggle repo
data = np.load("../input/mnist.npz")
X_train = data['x_train']
X_test  = data['x_test']
y_train = data['y_train']
y_test  = data['y_test']

X_train.shape
print("X: ",X_train)
print("Y: ",y_train)
# Proprocess data

# Flatten data 
X_train = X_train.reshape(60000, 28*28).astype('float32')
X_test = X_test.reshape(10000, 28*28).astype('float32')
# Scale grayscale data (256 levels) to 0 - 1  

X_train /= 255
X_test /= 255
# Set 10 prediction classes (0 - 10)

n_classes = 10
y_train = keras.utils.to_categorical(y_train, n_classes)
y_test = keras.utils.to_categorical(y_test, n_classes)
# Verify train and y-hat dataset

print(X_train.shape)
print(y_train.shape)
# Neural Network Model (784 inputs, 10 outputs)

model = Sequential()
model.add(Dense((64), activation = 'relu', input_shape=(784,)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense((64), activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense((10), activation = 'softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=["accuracy"])
# Train the model
model.fit(X_train, y_train, batch_size=128, epochs=100, verbose=1, validation_data=(X_test, y_test))
# Scoring
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1]*100,'%')
# Prediction shows expected result from the model. The feature with the highest value is the prediction
predictions = pd.DataFrame(model.predict(X_test))
predictions.head(10)
# We want to compare the prediction with the actual number of mnist
actual = pd.DataFrame(y_test)
actual.head(10)
# We convert both the prediction and actual value into mnist number 1-10 
Results = pd.concat([pd.Series(actual.idxmax(axis = 1)), pd.Series(predictions.idxmax(axis =1))], axis = 1)
Results.columns = ['Actual', 'Prediction']

# First 10 rows. Notice model prediction was wrong on the 8th row. 
Results.head(10)
# Get all the difference from dataframe. There are 370 out of 10,000 error = 3.70%, which is 
# consistent w/ the above model accuracy evaluation.

Results.loc[(Results['Actual'] - Results['Prediction']) != 0]
