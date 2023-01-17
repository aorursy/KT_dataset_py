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
# importing the necessary libraries

import numpy as np

import matplotlib.pyplot as plt

import os

import cv2

from keras.preprocessing.image import ImageDataGenerator

from keras.utils import normalize, to_categorical

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.callbacks import TensorBoard
# creating the training dataset

categories = ['open', 'close']

training_path = '../input/training_set/training_set'

training_set = []

for category in categories:

    path = os.path.join(training_path, category)

    label = categories.index(category)

    for img in os.listdir(path):

        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)

        if img_array is None:

            continue

        training_set.append([img_array, label])
# shuffle the training data

np.random.shuffle(training_set)
# splitting the training data into it's features and labels(X_train, y_train) and reshape them

X_train = []

y_train = []

for feature, label in training_set:

    X_train.append(feature)

    y_train.append(label)    

X_train = np.array(X_train).reshape((-1, 24, 24, 1))

y_train = np.array(y_train)
# creating the test dataset

categories = ['open', 'close']

test_path = '../input/test_set/test_set'

test_set = []

for category in categories:

    path = os.path.join(test_path, category)

    label = categories.index(category)

    for img in os.listdir(path):

        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)

        if img_array is None:

            continue

        test_set.append([img_array, label])
# shuffle the test set

np.random.shuffle(test_set)
# splitting the set data into it's features and labels(X_train, y_train) and reshape them

X_test = []

y_test = []

for feature, label in training_set:

    X_test.append(feature)

    y_test.append(label)

X_test = np.array(X_test).reshape(-1, 24, 24, 1)

y_test = np.array(y_test)
# normalizing the features(X_train, X_test)

X_train = normalize(X_train, axis = 1)

X_test = normalize(X_test, axis = 1)
# Converting the labels(y_train, y_test) to one hot encoder

num_classes = 2

y_train = to_categorical(y_train, num_classes = num_classes)

y_test = to_categorical(y_test, num_classes = num_classes)
# creating the CNN model

model = Sequential()

model.add(Conv2D(32, (5, 5), input_shape = (24, 24, 1), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(32, (3, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())

model.add(Dense(128, activation = 'relu'))

model.add(Dense(2, activation = 'softmax'))

model.summary()
# creating tensorboard

import time

name = "open-close-eye-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir = "logs/{}".format(name))
# compiling the model

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# fitting the training data to the model

history = model.fit(X_train, y_train, epochs = 5, validation_split = 0.1, callbacks = [tensorboard])
# visualizing the training and evaluation accuracy

plt.plot(np.arange(5), history.history['acc'])

plt.plot(np.arange(5), history.history['val_acc'])

plt.title('accuracy vs epochs')

plt.xlabel('epochs')

plt.ylabel('accuracy')

plt.legend(['training accuracy', 'validation accuracy'])

plt.show()
# Test set evalutaion

test_loss, test_accuracy = model.evaluate(X_test, y_test)

print(f'test loss: {test_loss:.3}')

print(f'test accuracy: {test_accuracy:.3}')
# make prediction with the trained model

sample_image = 100

prediction = np.argmax(model.predict(X_test[None, sample_image]))

actual = np.argmax(y_test[None, sample_image])

print("prediction: ", categories[prediction])

print("actual value: ", categories[actual])