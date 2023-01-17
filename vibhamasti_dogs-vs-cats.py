# Import libraries

import numpy as np

import keras

import cv2



import matplotlib.pyplot as plt

%matplotlib inline 



import os        # To see our directory

import random

import gc        # Garbage collector
train_dir = "../input/dogs-vs-cats-redux-kernels-edition/train"

test_dir = "../input/dogs-vs-cats-redux-kernels-edition/test"



train_cats = ['../input/dogs-vs-cats-redux-kernels-edition/train/{}'.format(i) for i in os.listdir(train_dir) if 'cat' in i]

train_dogs = ['../input/dogs-vs-cats-redux-kernels-edition/train/{}'.format(i) for i in os.listdir(train_dir) if 'dog' in i]



train_images = train_cats[:2000] + train_dogs[:2000]

random.shuffle(train_images)





del train_cats, train_dogs

gc.collect()



#import matplotlib.image as mpimg

#for ima in train_images[0:5]:

#    img = mpimg.imread(ima)

#    imgplot = plt.imshow(img)

#    plt.show()
# Image dimensions



nrows = 200

ncolumns = 200

# RGB are 3 channels

# For greyscale, 1 channel

channels = 3
def read_and_process_image(list_of_images):

    # X is an array of resized images

    # y is an array of labels

    

    X = []

    y = []

    

    for image in list_of_images:

        # Read the image

        X.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (nrows, ncolumns), interpolation=cv2.INTER_CUBIC))

        

        # Get the labels

        if 'dog' in image:

            y.append(1)

        elif 'cat' in image:

            y.append(0)

            

        return X, y
X, y = read_and_process_image(train_images)



del train_images

gc.collect()



X = np.array(X)

y = np.array(y)
from sklearn.model_selection import train_test_split



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, train_size=0.80, random_state=2)



print("Shape of train images is:", X_train.shape)

print("Shape of validation images is:", X_val.shape)

print("Shape of train labels is:", y_train.shape)

print("Shape of validation labels is:", y_val.shape)
# Built and fit CNN





model = keras.models.Sequential()



model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)))

model.add(Flatten())

model.add(Dense(2, activation='softmax'))



model.compile(Adam(lr=0.001), loss=categorical_crossentropy, metrics=['accuracy'])

model.fit_generator(train_images, steps_per_epoch=14, epochs=5, verbose=2)

          