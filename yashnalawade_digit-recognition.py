import numpy

import keras

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

import numpy as np 

import pandas as pd 

import os

from sklearn.model_selection import train_test_split

from matplotlib import pyplot



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

x= train.iloc[:, 1:].values  

y= train.iloc[:, 0].values  

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state=0)  

test.shape
epochs = 10

batch_size = 128

num_classes = 10

input_shape = (28, 28, 1)

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)



y_train = keras.utils.to_categorical(y_train, num_classes)

y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')

x_test = x_test.astype('float32')

x_train /= 255

x_test /= 255

print('x_train shape:', x_train.shape)

print(x_train.shape[0], 'train samples')

print(x_test.shape[0], 'test samples')
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])

hist = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))

print("The model has successfully trained")

model.save('mnist.h5')

print("Saving the model as mnist.h5")
test_data = test.to_numpy().reshape((-1,28,28,1))

test_data.shape

predictions=model.predict(test_data)

predictions=numpy.argmax(predictions, axis=1)
submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')

submission['Label'] = predictions

submission.to_csv("submission.csv", index=False, header=True)

print("Your submission was successfully saved!")