# basic lib

import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# LIME lib

import lime

from lime import lime_image

from skimage.segmentation import mark_boundaries



# CNN lib

from keras.models import Sequential

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.layers.core import Activation

from keras.layers.core import Flatten

from keras.layers.core import Dropout

from keras.layers.core import Dense

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import RMSprop

from keras import preprocessing



# warnings

import warnings

warnings.filterwarnings('ignore') 
# import data



path_cats = []

train_path_cats = '../input/training_set/training_set/cats'

for path in os.listdir(train_path_cats):

    if '.jpg' in path:

        path_cats.append(os.path.join(train_path_cats, path))

path_dogs = []

train_path_dogs = '../input/training_set/training_set/dogs'

for path in os.listdir(train_path_dogs):

    if '.jpg' in path:

        path_dogs.append(os.path.join(train_path_dogs, path))

len(path_dogs), len(path_cats)





# load training set

training_set = np.zeros((6000, 150, 150, 3), dtype='float32')

for i in range(6000):

    if i < 3000:

        path = path_dogs[i]

        img = preprocessing.image.load_img(path, target_size=(150, 150))

        training_set[i] = preprocessing.image.img_to_array(img)

    else:

        path = path_cats[i - 3000]

        img = preprocessing.image.load_img(path, target_size=(150, 150))

        training_set[i] = preprocessing.image.img_to_array(img)

        

# load validation set

validation_set = np.zeros((2000, 150, 150, 3), dtype='float32')

for i in range(2000):

    if i < 1000:

        path = path_dogs[i + 3000]

        img = preprocessing.image.load_img(path, target_size=(150, 150))

        validation_set[i] = preprocessing.image.img_to_array(img)

    else:

        path = path_cats[i + 2000]

        img = preprocessing.image.load_img(path, target_size=(150, 150))

        validation_set[i] = preprocessing.image.img_to_array(img)
# plot show data



plt.figure(figsize = (12,10))

row, colums = 4, 4

for i in range(16):  

    plt.subplot(colums, row, i+1)

    image = training_set[i].astype(np.uint8)

    plt.imshow(image)

plt.show()
# build CNN



model = Sequential()

model.add(Conv2D(32, (5, 5), activation='relu',

                       input_shape=(150, 150, 3)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu',

                       input_shape=(150, 150, 3)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',

             optimizer=RMSprop(lr=1e-4),

             metrics=['acc'])



model.summary()
# make target tensor

train_labels = np.zeros((3000,))

train_labels = np.concatenate((train_labels, np.ones((3000,))))

validation_labels = np.zeros((1000,))

validation_labels = np.concatenate((validation_labels, np.ones((1000,))))



train_datagen = preprocessing.image.ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow(

    training_set,

    train_labels,

    batch_size=32)



validation_generator = train_datagen.flow(

    validation_set,

    validation_labels,

    batch_size=32)
# fit the model



history = model.fit_generator(

    train_generator,

    steps_per_epoch=100,

    epochs= 8,

    validation_steps=50,

    validation_data=validation_generator,

)
# create lime ImageExplainer

explainer = lime_image.LimeImageExplainer()
image_number = 2





image = training_set[image_number].astype(np.uint8)

image * 1/255

explanation = explainer.explain_instance(image, model.predict, top_labels=5, hide_color=0, num_samples=1000)

temp, mask = explanation.get_image_and_mask(0, positive_only=False, num_features=5, hide_rest=False)

plt.imshow(mark_boundaries(temp / 2 + 0.5, mask).astype(np.uint8))
# Check number of images 



plt.figure(figsize = (12,10))

row, colums = 3, 3

for i in range(9):  

    plt.subplot(colums, row, i+1)

    image = training_set[i].astype(np.uint8)

    image * 1/255

    explanation = explainer.explain_instance(image, model.predict, top_labels=5, hide_color=0, num_samples=1000)

    temp, mask = explanation.get_image_and_mask(0, positive_only=False, num_features=5, hide_rest=False)

    plt.imshow(mark_boundaries(temp / 2 + 0.5, mask).astype(np.uint8))

plt.show()
image_number = 20





image = training_set[image_number].astype(np.uint8)

image * 1/255

explanation = explainer.explain_instance(image, model.predict, top_labels=5, hide_color=0, num_samples=1000)

temp, mask = explanation.get_image_and_mask(0, positive_only=False, num_features=5, hide_rest=True)

plt.imshow(mark_boundaries(temp / 2 + 0.5, mask).astype(np.uint8))
image_number = 34



image = training_set[image_number].astype(np.uint8)

image * 1/255

explanation = explainer.explain_instance(image, model.predict, top_labels=5, hide_color=0, num_samples=1000)

temp, mask = explanation.get_image_and_mask(0, positive_only=True, num_features=5, hide_rest=False)

plt.imshow(mark_boundaries(temp / 2 + 0.5, mask).astype(np.uint8))
image_number = 62



image = training_set[image_number].astype(np.uint8)

image * 1/255

explanation = explainer.explain_instance(image, model.predict, top_labels=5, hide_color=0, num_samples=1000)

temp, mask = explanation.get_image_and_mask(0, positive_only=False, num_features=20, hide_rest=False)

plt.imshow(mark_boundaries(temp / 2 + 0.5, mask).astype(np.uint8))