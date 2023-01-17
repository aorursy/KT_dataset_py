import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

import csv



#Load training and testing data as csv file

training_data = pd.read_csv("/kaggle/input/emnist/emnist-mnist-train.csv").to_numpy()

testing_data = pd.read_csv("/kaggle/input/emnist/emnist-mnist-test.csv").to_numpy()

print(training_data.shape)
#Format he data into its Xs and Ys

training_Xs = []

training_Ys = []

testing_Xs  = []

testing_Ys  = []

#Get the training Xs and Ys

for img in training_data:

    training_Xs.append(img[1:])

    training_Ys.append(img[0])



#Get testing Xs and Ys

for img in testing_data:

    testing_Xs.append(img[1:])

    testing_Ys.append(img[0])
#Normalize the input arrays (0 -> 0 | 255 -> 1)

training_Xs /= np.max(np.abs(training_Xs))

testing_Xs /= np.max(np.abs(testing_Xs))



#Convert the data to np arrays + reshape it

training_Xs = np.asarray(training_Xs)

training_Xs = training_Xs.reshape((-1, 28, 28, 1))



testing_Xs  = np.asarray(testing_Xs)

testing_Xs  = testing_Xs.reshape((-1, 28, 28, 1))



print(training_Xs.shape)
#Convert Ys to 1-hot arrays

new_Ys = []

for lable in training_Ys:

    a = np.zeros(10)

    for i in range(10):

        if i == lable:

            a[i] = 1

    new_Ys.append([])

    new_Ys[len(new_Ys)-1] = a



training_Ys = np.asarray(new_Ys)



new_Ys = []

for lable in testing_Ys:

    a = np.zeros(10)

    for i in range(10):

        if i == lable:

            a[i] = 1

    new_Ys.append([])

    new_Ys[len(new_Ys)-1] = a



testing_Ys = np.asarray(new_Ys)

print(training_Ys[0])
import keras

from keras.models import Sequential

from keras.layers import Dense, Flatten, MaxPooling2D, Conv2D



classifier = Sequential()

classifier.add(Conv2D(32, (2, 2), activation="relu", input_shape=(28, 28, 1)))

classifier.add(MaxPooling2D())

classifier.add(Conv2D(32, (2, 2), activation="relu"))

classifier.add(MaxPooling2D())

classifier.add(Flatten())



classifier.add(Dense(units=128, activation="relu"))

classifier.add(Dense(units=10, activation="relu"))



classifier.compile(loss="mean_squared_error", optimizer="sgd")



classifier.summary()
#Train the classifier on the data

BATCH_SIZE = 64

RUN_COUNT = 10

a = 0



for i in range(RUN_COUNT):

    classifier.fit(training_Xs, training_Ys, BATCH_SIZE, verbose=1, shuffle=True)

    #Show Example

    if i % 1 == 0:

        #Show an example prediction + real value

        d = testing_Xs[a]

        d = np.reshape(d, (1, 28, 28, 1))



        p = classifier.predict(d)[0]

        p = np.where(p == np.amax(p))[0][0]

        print("Epoch: " + (str)(i) + " | Prediction: " + (str)(p))

        plt.imshow(np.reshape(d, (28, 28)))

        plt.show()

        a += 1

        