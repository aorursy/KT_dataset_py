# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import matplotlib.pyplot as plt

import os

import cv2

import random



import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Activation, Conv2D, Dropout, Flatten, MaxPooling2D, BatchNormalization, SeparableConv2D

from tensorflow.keras.optimizers import SGD



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory





#for dirname, _, filenames in os.walk('/kaggle/input/chest-xray-pneumonia/chest_xray/train'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
print(os.listdir("../input/chest-xray-pneumonia/chest_xray"))



dirs = "../input/chest-xray-pneumonia/chest_xray/train"



categories = ["NORMAL", "PNEUMONIA"]



for category in categories:

    path = os.path.join(dirs, category)

    for image in os.listdir(path):

        img_array = cv2.imread(os.path.join(path,image),  cv2.IMREAD_GRAYSCALE)

        plt.imshow(img_array, cmap="gray")

        plt.show()

        break

    

    print(img_array)
'''

Initial imageSize attempt was 800 * 800 sized images, but that was too much computation for this laptop.

in the kaggle notebook I can continue to use this!

'''

imageSize = 300



for category in categories:

    path = os.path.join(dirs, category)

    for image in os.listdir(path):

        img_array = cv2.imread(os.path.join(path,image),  cv2.IMREAD_GRAYSCALE)

        resized = cv2.resize(img_array, (imageSize, imageSize))

        plt.imshow(resized, cmap="gray")

        plt.show()

        break

    

    print(img_array)
train = []



#My labels on the training data will be done with the indexing of our categories above 0=Normal, 1 = Pneumonia.



for category in categories:

    path = os.path.join(dirs, category)

    classification = categories.index(category)

    for image in os.listdir(path):

        img_array = cv2.imread(os.path.join(path,image),  cv2.IMREAD_GRAYSCALE)

        resized = cv2.resize(img_array, (imageSize, imageSize))

        train.append([resized, classification])

        

print("Printing first image as an array")

print(train[0])



print("Number of images in training set:", len(train))
test = []

dirs = "../input/chest-xray-pneumonia/chest_xray/test"





for category in categories:

    path = os.path.join(dirs, category)

    classification = categories.index(category)

    for image in os.listdir(path):

        img_array = cv2.imread(os.path.join(path,image),  cv2.IMREAD_GRAYSCALE)

        resized = cv2.resize(img_array, (imageSize, imageSize))

        test.append([resized, classification])

print("Printing the first image as an array")  

print(test[0])

print("Number of images in our test set", len(test))
val = []

dirs = "../input/chest-xray-pneumonia/chest_xray/val"







for category in categories:

    path = os.path.join(dirs, category)

    classification = categories.index(category)

    for image in os.listdir(path):

        img_array = cv2.imread(os.path.join(path,image),  cv2.IMREAD_GRAYSCALE)

        resized = cv2.resize(img_array, (imageSize, imageSize))

        val.append([resized, classification])

print("Printing first image as an array")    

print(val[0])

print("Number of images in validation set", len(val))
random.shuffle(train)

random.shuffle(val)

random.shuffle(test)

for img in train[:10]:

    print(img[1])
train_X = [] 

train_y = [] 



test_X = [] 

test_y = [] 



val_X = []

val_y = []



for entity, label in train:

    train_X.append(entity)

    train_y.append(label)

    

for entity, label in test:

    test_X.append(entity)

    test_y.append(label)



for entity, label in val:

    val_X.append(entity)

    val_y.append(label)



print("Done!")
plt.imshow(train_X[0], cmap="gray")

print(train_X[0])

#Checking to see type conversion with tensorflow

#print(type(train_X))
train_X = tf.keras.utils.normalize(train_X, axis=1)

test_X = tf.keras.utils.normalize(test_X, axis=1)

val_X = tf.keras.utils.normalize(val_X, axis=1)

print("done")

plt.imshow(train_X[0], cmap="gray")

print(train_X[0])

print(type(train_X))
print(type(train_y))





train_y = np.array(train_y)



test_y = np.array(test_y)



val_y= np.array(val_y)



print(type(train_y))

print("Done")

# -1 is the catch all taht creates the fourth parameter that the neural network needs, the middle two are the image Length and width

# 1 is the color mapping , we are grayscale so 1 color pallete is necesary, RGB would be 3.

train_X = np.array(train_X).reshape(-1, imageSize, imageSize, 1)

test_X = np.array(test_X).reshape(-1, imageSize, imageSize, 1)

val_X = np.array(val_X).reshape(-1, imageSize, imageSize, 1)

print("Done!")
#Checking to ensure normalization worked on all data partitions (it did)

#print("Train")

#print(train_X[0])

#print(len(train_X))

#print("Test")

#test_X

#len(test_X)

#print("Val")

#val_X[0]
train_X.shape
print(len(train_y))
# Model type is sequential, standard here it seems.

model = Sequential()

# Optimizer would be 'adam' or any of the other default given ones

# but using the actual SGD Stochastic Gradient Descent optimizer allows us to tweak learning rate and momentum.

opt = SGD(lr=0.01, momentum=0.9)





# Initial stride larger and k_size larger too to get "larger" understanding of image

# Starting with conv2d and transitioning to separableconv2, the computations are the same(?) but the separable is far more efficient

# https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d

# Continue using BatchNorm as network gets deeper. Used maxpooling2D as my dimensionality reduction method.



model.add(Conv2D(64, (3,3), activation='relu', padding='same',input_shape=(imageSize,imageSize,1)))

model.add(BatchNormalization())

model.add(Conv2D(64, (3,3), activation='relu', padding='same'))

model.add(MaxPooling2D((2,2)))



#Begin Smaller Sep.Conv2D Rectified Linear Unit activation throughout until the end

model.add(SeparableConv2D(128, (3,3), activation='relu', padding='same'))

model.add(BatchNormalization())

model.add(SeparableConv2D(128, (3,3), activation='relu', padding='same'))

model.add(MaxPooling2D((2,2)))



model.add(SeparableConv2D(256, (3,3), activation='relu', padding='same'))

model.add(BatchNormalization())

model.add(SeparableConv2D(256, (3,3), activation='relu', padding='same'))

model.add(MaxPooling2D((2,2)))



model.add(SeparableConv2D(512, (3,3), activation='relu', padding='same'))

model.add(BatchNormalization())

model.add(SeparableConv2D(512, (3,3), activation='relu', padding='same'))

model.add(MaxPooling2D((2,2)))

# create 1D array and some fully connected layers.

model.add(Flatten())

model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.3,))

# Finally conluding on 1 of 2 possible outputs we get the classification, using sigmoid activation.

# See http://cs231n.github.io/neural-networks-1/#actfun "Common activation functions"

model.add(Dense(1, activation='sigmoid'))

    



'''

Finally finishing this up with a Dense layer to give us our result of 0 or 1.

In order to use a dense layer the data must be 1d so we use a Flatten layer to "flatten" the images.

'''

#The flatten into a full connected Layer and then finally, the classification of our image.







model.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy'])







model.fit(train_X, train_y, batch_size=30, epochs=10, validation_data=(val_X, val_y))

prediction = model.evaluate(test_X, test_y)

print()