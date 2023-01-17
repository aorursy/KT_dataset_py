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



from keras.applications import VGG16

from keras.preprocessing.image import ImageDataGenerator



vgg_conv = VGG16(weights='imagenet',

                  include_top=False,

                  input_shape=(224, 224, 3))
from keras.applications import VGG16

image_size = 224

#Load the VGG model

vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))



# Freeze all the layers

for layer in vgg_conv.layers[:]:

    layer.trainable = False



# Check the trainable status of the individual layers

for layer in vgg_conv.layers:

    print(layer, layer.trainable)





from keras import models

from keras import layers

from keras import optimizers



# Create the model

model = models.Sequential()



# Add the vgg convolutional base model

model.add(vgg_conv)



# Add new layers

model.add(layers.Flatten())

model.add(layers.Dense(1024, activation='relu'))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(95, activation='softmax'))



# Show a summary of the model. Check the number of trainable parameters

model.summary()
train_datagen = ImageDataGenerator(

        rescale=1./255,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True)



test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

        "../input/fruits-360_dataset/fruits-360/Training",

        target_size=(224, 224),

        batch_size=256,

        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(

        "../input/fruits-360_dataset/fruits-360/Test",

        target_size=(224, 224),

        batch_size=256,

        class_mode='categorical')
model.compile(loss='categorical_crossentropy',

              optimizer='rmsprop',

              metrics=['accuracy'])
model.fit_generator(

        train_generator,

        steps_per_epoch=100,

        epochs=10,

        validation_data=validation_generator,

        validation_steps=100)
fruit_indices = validation_generator.class_indices

def which_fruit(yhat, fruit_indices):

    inverted = argmax(yhat)

    for key in fruit_indices:

        if fruit_indices[key] == inverted:

            return key
ext = '../input/fruits-360_dataset/fruits-360/test-multiple_fruits'

from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array

from keras.applications.vgg16 import preprocess_input

from numpy import argmax

import matplotlib.pyplot as plt

for i in os.listdir(ext):

    # convert the image pixels to a numpy array

    image = load_img(ext+'/'+i, target_size=(224, 224))

    plt.imshow(image)

    image = img_to_array(image)

    # reshape data for the model

    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    

    # prepare the image for the VGG model

    image = preprocess_input(image)

    # predict the probability across all output classes

    yhat = model.predict(image)    

    print(i.split('_'), which_fruit(yhat, fruit_indices))