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
import pandas as pd
from tensorflow import keras
print(tf.__version__)
train = pd.read_csv("../input/train.csv")
print("train dataset shape is ", train.shape)
train.head()

# All pixel values - all rows and column 1 (pixel0) to column 785 (pixel 783)
x_train = (train.iloc[:,1:].values).astype('float32') 
# Take a look at x_train
x_train
# Labels - all rows and column 0
y_train = (train.iloc[:,0].values).astype('int32') 

# Take a look at y_train
y_train
test = pd.read_csv("../input/test.csv")
print("test dataset shape is ", test.shape)
test.head()
x_test = test.values.astype('float32')
x_test
num_classes = 10

# Normalize the input data
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Reshape input data from (28, 28) to (28, 28, 1)
w, h = 28, 28
x_train = x_train.reshape(-1, w, h, 1)
x_test = x_test.reshape(-1, w, h, 1)

# One-hot encode the labels
y_train = keras.utils.to_categorical(y_train, num_classes)

# Take a look at the dataset shape after conversion with keras.utils.to_categorical
print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)
model = keras.Sequential()

# Must define the input shape in the first layer of the neural network
model.add(keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(28,28,1))) 
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D(pool_size=2))
model.add(keras.layers.Dropout(0.3))

model.add(keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D(pool_size=2))
model.add(keras.layers.Dropout(0.3))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(10, activation='softmax'))

# Take a look at the model summary
model.summary()
model.compile(loss=keras.losses.categorical_crossentropy,
             optimizer=keras.optimizers.Adam(),
             metrics=['accuracy'])
model.fit(x_train,
         y_train,
         batch_size=64,
         epochs=40)
predictions = model.predict_classes(x_test, verbose=0)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("mnist_digi_recog.csv", index=False, header=True)