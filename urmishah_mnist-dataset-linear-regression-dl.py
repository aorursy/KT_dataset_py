# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#mnist dataset is available in tensorflow itself
from tensorflow.keras.datasets.mnist import load_data
(X_train, y_train), (X_test, y_test) = load_data()
print(X_train.shape)
print(X_test.shape)
#how one image looks like
X_train[0]
#maplotlib function which helps to display the image of the array of numbers
plt.matshow(X_train[0])
#crosscheck
y_train[0]
#data normalization
X_train = X_train/255
X_test = X_test/255

"""
Why divided by 255?
The pixel value lie in the range 0 - 255 representing the RGB (Red Green Blue) value. """
#let's check
X_train[0]
X_train.shape
#2d to 1d conversion
X_train_flattened = X_train.reshape(len(X_train), 28*28)
X_test_flattened = X_test.reshape(len(X_test), 28*28)
X_train_flattened.shape
#1. Define the model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from numpy.random import seed
import tensorflow
model = Sequential()
model.add(Dense(10, input_shape=(784,), activation='sigmoid'))
model.summary()
#2. Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#3. Fitting the model
model.fit(X_train_flattened, y_train, epochs=10)
#4. Evaluate the model
model.evaluate(X_test_flattened, y_test)
#5. Make Prediction
y_predicted = model.predict(X_test_flattened)
y_predicted[0]
#np.argmax finds a maximum element from an array and returns the index of it
np.argmax(y_predicted[0])
#check
plt.matshow(X_test[0])
