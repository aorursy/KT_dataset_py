# import libraries
import pandas as pd
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, InputLayer
from keras.optimizers import RMSprop
# read csv
# train.csv and test.csv NO NAIYOU WA ONAJI (label GA ARU KA NAI KA DAKE DE) NANODE
# INCHIKI SUREBA score 1.00000 MO KANTAN NO HAZU DAKEDO,
# Example TO SHITE machine learning DE predict SHITE MIRU.
df_train = pd.read_csv('../input/digit-recognizer/train.csv')
df_test  = pd.read_csv('../input/digit-recognizer/test.csv')

# show df_train
df_train
# convert dataframe to np.array
array_train = np.array(df_train)
array_test  = np.array(df_test)
# separate labels(y) and pixels(x) in training data
x_train = array_train[:, 1:]
y_train = array_train[:, 0]
x_test  = array_test

# rescale x
x_train = x_train.astype('float32')
x_test  = x_test .astype('float32')
x_train /= 255
x_test  /= 255

# convert labels(y) into one-hot vectors
y_train = keras.utils.to_categorical(y_train, 10)
# generate model
model = Sequential()
model.add(InputLayer(input_shape = (784, )))
model.add(Dense(128, activation = 'sigmoid'))
model.add(Dense(128, activation = 'sigmoid'))
model.add(Dense(10, activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
# learn
epochs = 100
batch_size = 128
validation_split = 0.2
history = model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, validation_split = validation_split, verbose = 0) #this line takes a few minutes
# predict labels for the test data
y_test = model.predict(x_test, verbose = 1)
# convert one-hot vectors into integers
y_test = np.argmax(y_test, axis = 1)
# output data in the designated form
imageId = np.arange(1, y_test.shape[0] + 1)
df_output = pd.DataFrame({'ImageId': imageId,
                          'Label'  :y_test})
df_output.to_csv('/kaggle/working/output.csv', index = False)
df_output

#-> download output.csv and submit!