import os

import cv2

import random

import numpy as np

import matplotlib.pyplot as plt



#Load the data fram the images into arrays

Xs = []

Ys = 0

lables = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        path = os.path.join(dirname, filename)

        #Load the image using openCV2

        Xs.append(cv2.imread(path, cv2.IMREAD_GRAYSCALE))

        #Use the file's name to get it's content (-.png)

        name = filename[:-4]

        lables.append(name)            



images = Xs;

Xs = np.asarray(Xs)

Xs = np.reshape(Xs, (len(Xs), 50, 200, 1))

lables = np.asarray(lables)



#Print the value of 1 pixel and the name of it's coresponding captcha

print(lables[0])

print(Xs[0][0][0])
char_dict = {}

num_dict = {}

i = 0

counter = 0



Ys = np.zeros((len(Xs), 5, 19))

#Format the data

for y in lables:

    #See if the character is in the dictionary

    for j in range(5):

        c = y[j]

        if not (c in char_dict):

            #Add the name to the dictionarys

            char_dict[c] = i

            num_dict[i] = c

            #Convert it to the array

            i += 1

        index = char_dict.get(c)

        Ys[counter][j][index] = 1

    counter += 1



Ys = np.asarray(Ys)



print(lables[10])

print(Ys[10])
#Construct the network 

    #Input Shape: 50 x 200 x 1(color value)

    #Output Shape: ([19 characters] * 5 characters) = 95

import keras

from keras.models import Sequential

from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Reshape



classifier = Sequential()

#First we pass the data through some convolutional and maxPooling layers

classifier.add(Conv2D(filters=1, kernel_size=(2, 2), activation="relu", input_shape=(50, 200, 1)))

classifier.add(MaxPooling2D())

classifier.add(Conv2D(filters=1, kernel_size=(2, 2), activation="relu"))

classifier.add(MaxPooling2D())

classifier.add(Flatten())

classifier.add(Dense(units=256, activation="relu"))

classifier.add(Dense(units=95, activation="sigmoid"))

#Reshaped to fit the shape of the Ys

classifier.add(Reshape(target_shape=(5, 19)))



classifier.compile(optimizer="adam", loss="mean_squared_error")
#Now that I have the classifier, train it on the dataset

train_Xs = Xs[101:]

testing_Xs = Xs[:100]

training_Ys = Ys[101:]

testing_Ys = Ys[:100]

BATCH_SIZE = 64

EPOCHS = 10

PROOF_RATE = 10



for i in range(100):

    classifier.fit(train_Xs, training_Ys, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=0, shuffle=True)

    #Take a random image and check in on the network

    index = random.randint(0, len(testing_Xs)-1)

    test = testing_Xs[index]

    prediction = classifier.predict(np.reshape(test, (1, 50, 200, 1)))

    #Convert the prediction to a lable

    pred_lable = ", Predicted Lable: "

    true_lable = ", Real Lable: " + (str)(lables[index])

    for c in prediction[0]:

        hot = np.amax(c)

        a = np.where(c == hot)

        char = num_dict[a[0][0]]

        pred_lable += (str)(char)

    

    print("Epoch: " + (str)(i+1) + pred_lable + true_lable)

    print(classifier.evaluate(testing_Xs, testing_Ys))

    if i % PROOF_RATE == 0:

        plt.imshow(images[index])

        plt.show()

        

        