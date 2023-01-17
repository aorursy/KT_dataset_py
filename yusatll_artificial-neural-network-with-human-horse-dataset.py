# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.image as img

import cv2

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

i=0

for dirname, _, filenames in os.walk('/kaggle/input'):

    print(dirname)

   ## for filename in filenames:

        #print(filename) # png names

        

        #print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_horses_path = ("/kaggle/input/horses-or-humans-dataset/horse-or-human/horse-or-human/train/horses/")

train_humans_path = ("/kaggle/input/horses-or-humans-dataset/horse-or-human/horse-or-human/train/humans/")

validation_horses_path = ("/kaggle/input/horses-or-humans-dataset/horse-or-human/horse-or-human/validation/horses/")

validation_humans_path = ("/kaggle/input/horses-or-humans-dataset/horse-or-human/horse-or-human/validation/humans/")
example = img.imread("/kaggle/input/horses-or-humans-dataset/horse-or-human/horse-or-human/train/horses/horse01-6.png")

example = cv2.cvtColor(example, cv2.COLOR_BGR2GRAY) # this is for create a 2D image matrix.

print(example.dtype)

print(example.shape)

plt.imshow(example)

plt.axis('off')

plt.show()
train_horse = []

train_human = []

valid_horse = []

valid_human = []



# read train horses image dataset

for png in os.listdir(train_horses_path):

    imageread = img.imread(train_horses_path+png)

    imageread = cv2.cvtColor(imageread, cv2.COLOR_BGR2GRAY)

    train_horse.append(imageread)

    #print(imgread.shape) # (300, 300)



print(len(train_horse) , "horses images found in train folder.")
# read train humans image dataset

for png in os.listdir(train_humans_path):

    imageread = img.imread(train_humans_path+png)

    imageread = cv2.cvtColor(imageread, cv2.COLOR_BGR2GRAY)

    train_human.append(imageread)

    # print(imgread.shape) # (300, 300)

    

print(len(train_human) , "humans images found in train folder.")   
# concatenate our train images dataset

all_train_images = np.concatenate((train_human, train_horse), axis = 0)

print("All train images :",all_train_images.shape) # (1027, 300, 300)  -> this means we have 1027 images and these image 300x300 pixels.

# we do not need this. Only find the total number of images.
# read horses validation datasets.

for png in os.listdir(validation_horses_path):

    imageread = img.imread(validation_horses_path+png)

    imageread = cv2.cvtColor(imageread, cv2.COLOR_BGR2GRAY)

    valid_horse.append(imageread)

    #print(imgread.shape) # (300, 300)

print(len(valid_horse), "horses images data for validation.")
# read humans validation datasets

for png in os.listdir(validation_humans_path):

    imageread = img.imread(validation_humans_path+png)

    imageread = cv2.cvtColor(imageread, cv2.COLOR_BGR2GRAY)

    valid_human.append(imageread)

    #print(imgread.shape) # (300, 300)

print(len(valid_horse), "humans images data for validation.")
# concatenate our validation images dataset

all_valid_images = np.concatenate((valid_horse, valid_human), axis = 0)

print("All validation images :",all_valid_images.shape)  # (256, 300, 300) -> we have 256 images for validation.

# we do not need this. Only find the total number of images.
# all images

x_data = np.concatenate((train_human, valid_human, train_horse, valid_horse), axis=0)

print(x_data.shape[0],"images have ",x_data.shape[1],"x",x_data.shape[2],"pixels.")
# We create our classify data. 1 for human and 0 for horses. 

zero = np.zeros(len(train_horse) + len(valid_horse)) # all horse images

one = np.ones(len(train_human) + len(valid_human))   # all human images

print("Number of humans images :", one.size)

print("Number of horses images :", zero.size)
# Y data

y = np.concatenate((zero, one), axis= 0).reshape(-1,1)

print(y.shape)

# [ 0 - 627] -> 0 -> horse

# [628-1282] -> 1 -> human
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_data, y, test_size = 0.3, random_state = 42)

# x_train.shape -> (1026, 300, 300)



number_of_train = x_train.shape[0]

number_of_test  = x_test.shape[0]



print("Number of train :", number_of_train)

print("Number of test :", number_of_test)
# flatten our data

x_train_flatten = x_train.reshape(number_of_train, x_train.shape[1] * x_train.shape[2])  # 898, 300*300

x_test_flatten = x_test.reshape(number_of_test, x_test.shape[1] * x_test.shape[2])       # 385, 300*300



print("X train Flatten : ",x_train_flatten.shape)

print("X test Flatten : ",x_test_flatten.shape)

x_train = x_train_flatten

x_test = x_test_flatten
# import Keras and layers libraries

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score

from keras.models import Sequential

from keras.layers import Dense
# this function is our classifier function. we make hidden layers in this funcion.

def build_classifier():

    classifier = Sequential()

    classifier.add(Dense(units = 50, kernel_initializer = 'uniform', activation = 'relu', input_dim = x_train.shape[1])) # first hidden layer

    classifier.add(Dense(units = 40, kernel_initializer = 'uniform', activation = 'relu'))   # second hidden layer

    classifier.add(Dense(units = 30, kernel_initializer = 'uniform', activation = 'relu'))   # third hidden layer

    classifier.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))

    #classifier.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))

    #classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu'))

    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid')) # last (output) layer

    

    classifier.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])

    return classifier
cf = KerasClassifier(build_fn=build_classifier, epochs = 100) # epochs = number of iteration

accuracies = cross_val_score(estimator=cf, X = x_train, y = y_train, cv = 3)

maks = accuracies.max()

variance = accuracies.std()

mean = accuracies.mean()



print("Accuracy max : ", maks)

print("Accuracy variance : ", variance)

print("Accuracy mean : ", mean)