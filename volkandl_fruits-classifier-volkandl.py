#Libraries



from keras.layers import Input, Conv2D, MaxPooling2D

from keras.layers import Dense, Flatten

from keras.models import Model

from keras.applications.vgg16 import decode_predictions

from keras.applications.vgg16 import preprocess_input

from keras.preprocessing import image

import matplotlib.pyplot as plt 

from PIL import Image 

import seaborn as sns

import pandas as pd 

import numpy as np 

import os 

from keras.applications import VGG19

from os import listdir, makedirs

from os.path import join, exists, expanduser

import tensorflow as tf

from sklearn.metrics import confusion_matrix

from numpy import newaxis

import cv2

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D,Dense,Activation,Dropout,Flatten,BatchNormalization

import matplotlib.pyplot as plt

from glob import glob

from keras.applications import VGG19

from skimage.feature import local_binary_pattern

from tensorflow.keras.optimizers import Adam

import numpy as np

from tensorflow.keras.regularizers import l2

from keras import applications

from keras.preprocessing.image import ImageDataGenerator

from keras import optimizers

from keras.models import Sequential, Model

from keras.layers import Dense, GlobalAveragePooling2D

from keras.layers import Activation, Dropout, Flatten, Dense

from keras import backend as K

import tensorflow as tf

import matplotlib.pyplot as plt
#Batch for Keras ImageGenerator is 16, img dimensions are 100,100

img_width, img_height = 100, 100 #

train_data_dir = '../input/fruits/fruits-360/Training/'

validation_data_dir = '../input/fruits/fruits-360/Test/'



batch_size = 16





train_datagen = ImageDataGenerator(

    rescale=1. / 255,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True)



test_datagen = ImageDataGenerator(rescale=1. / 255)





train_generator = train_datagen.flow_from_directory(

    train_data_dir,

    target_size=(img_height, img_width),

    batch_size=batch_size,

    class_mode='categorical')



validation_generator = test_datagen.flow_from_directory(

    validation_data_dir,

    target_size=(img_height, img_width),

    batch_size=batch_size,

    class_mode='categorical')



plt.figure(figsize=(5,5))

plt.axis('off')

plt.imshow(np.array(Image.open("../input/fruits/fruits-360/Training/Banana/120_100.jpg")))





plt.figure(figsize=(5,5))

plt.axis('off')

plt.imshow(np.array(Image.open("../input/fruits/fruits-360/Training/Avocado/100_100.jpg")))

print("shape of the banana",(np.array(Image.open("../input/fruits/fruits-360/Training/Banana/120_100.jpg"))).shape)

print("shape of the avocado",(np.array(Image.open("../input/fruits/fruits-360/Training/Avocado/100_100.jpg"))).shape)
numberOfClass=131

batch_size = 8

epochs=32





vgg = VGG19(include_top= False, weights = "imagenet", input_shape=(100,100,3))

vgg_layer_list = vgg.layers



model = Sequential ()



for layer in vgg_layer_list:

    model.add(layer)

    

for layer in model.layers:

    layer.trainable = False

    

    

model.add(Flatten())

model.add(Dense(1024))

model.add(Activation('relu'))

model.add(Dropout(0.2))



model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dropout(0.2))



model.add(Dense(numberOfClass))

model.add(Activation('softmax'))

model.summary()



opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)



model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])











hist = model.fit_generator(

        generator=train_generator,

        steps_per_epoch = 1500,

        epochs=epochs,

        validation_data=validation_generator,

        validation_steps = 220,

        shuffle=True)







# model plot



print(hist.history.keys())



plt.plot(hist.history["loss"], label ="Train Loss")

plt.plot(hist.history["val_loss"], label ="Validation Loss")

plt.legend()

plt.show()

plt.figure()

plt.plot(hist.history["accuracy"], label ="Train accuracy")

plt.plot(hist.history["val_accuracy"], label ="Validation accuracy")

plt.legend()

plt.show()
inception_base = applications.ResNet50(weights='imagenet', include_top=False)



x = inception_base.output

x = GlobalAveragePooling2D()(x)



x = Dense(512, activation='relu')(x)



predictions = Dense(131, activation='softmax')(x)





inception_transfer = Model(inputs=inception_base.input, outputs=predictions)

inception_transfer.compile(loss='categorical_crossentropy',

              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),

              metrics=['accuracy'])





with tf.device("/device:GPU:0"):

    history_pretrained = inception_transfer.fit_generator(

    train_generator,

    epochs=5, shuffle = True, verbose = 1, validation_data = validation_generator)