import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import shutil

import os

from PIL import Image

def make_directory(dir_path):

    if os.path.exists(dir_path):

        shutil.rmtree(dir_path)

    os.makedirs(dir_path)

    print('folder ', dir_path, ' is created')
import zipfile



DATASET_PATH = '../output/dogs-vs-cats'

DATASET_PATH_TMP = '../output/dogs-vs-cats/tmp'
make_directory(DATASET_PATH)

make_directory(DATASET_PATH_TMP)



# extract train data

with zipfile.ZipFile('../input/dogs-vs-cats/train.zip', 'r') as zip_ref:

    zip_ref.extractall(DATASET_PATH_TMP)



# extract test data

with zipfile.ZipFile('../input/dogs-vs-cats/test1.zip', 'r') as zip_ref:

    zip_ref.extractall(DATASET_PATH)



print('done')
from shutil import copyfile

import matplotlib.pyplot as plt

import tensorflow as tf

from keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.preprocessing.image import ImageDataGenerator

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',

                                            patience=2,

                                            verbose=1,

                                            factor=0.5,

                                            min_lr=0.00001)

callbacks = [learning_rate_reduction]

# All files are here including train.zip and test1.zip

base_dir = DATASET_PATH

# This is where I have extracted train.zip, will copy from here to train/cats and train/dogs and to validation/cats 

# and validation/dogs

tmp_dir = os.path.join(base_dir, 'tmp/train')

# This is training folder

train_dir = os.path.join(base_dir, 'train')

# This is validation folder. We will copy from train 20% of the dogs to validation/dogs and 20% of cats to validation/Cats

# Validation is part of the development procss of the model, whether the data samples are assigned to validation set 

# dynamically through k-Fold Corss-Validation or it is a fixed set from the beginning of training till the end.

validation_dir = os.path.join(base_dir, 'validation')

# This is test folder, we extract test1.zip here. This is the 'Production' Dataset where you don't know the labels 

# of the data samples.

test_dir = os.path.join(base_dir, 'test1')

# Directory with our training cat/dog pictures

train_cats_dir = os.path.join(train_dir, 'cats')

train_dogs_dir = os.path.join(train_dir, 'dogs')

validation_cats_dir = os.path.join(validation_dir, 'cats')

validation_dogs_dir = os.path.join(validation_dir, 'dogs')
print('Creating folders ....')

make_directory(train_dir)

make_directory(train_cats_dir)

make_directory(train_dogs_dir)

make_directory(validation_dir)

make_directory(validation_cats_dir)

make_directory(validation_dogs_dir)

print('Done')
list_of_fnames = os.listdir(tmp_dir)

list_of_cats_fnames = [i for i in list_of_fnames if 'CAT' in i.upper()]

print('Found {0} CATS images in input folder tmp/train'.format(len(list_of_cats_fnames)))

list_of_dogs_fnames = [i for i in list_of_fnames if 'DOG' in i.upper()]

print('Found {0} DOGS images in input folder tmp/train'.format(len(list_of_dogs_fnames)))



np.random.shuffle(list_of_cats_fnames)

np.random.shuffle(list_of_dogs_fnames)
TOTAL_CATS = len(list_of_cats_fnames)

TOTAL_DOGS = len(list_of_dogs_fnames)



TRAIN_VALIDATION_SPLIT_AT = 0.8



BATCH_SIZE = 100

TARGET_SIZE = (128, 128)

NO_OF_EPOCHS = 25

EXPERIMENT_SIZE = 12500  # Size of the sample set per category, cats or doags.

# This is to control how many samples we want to experiment the model on.

# This helps to build the model incrementally by experimenting on smaller

# set size, train untill over fit, then to seek better performance we increase complexity of the network

# train again until we overfit, add more data, and so on untill we we make use of all data available.



print('\nDistributing images to \n {0} \n {1} \n {2} \n {3}'

      '\nsuch that {4}% of total number of images goes to training and \n'

      '{5}% goes to validation'.format(

    train_cats_dir, train_dogs_dir,

    validation_cats_dir, validation_dogs_dir,

    round(TRAIN_VALIDATION_SPLIT_AT * 100),

    round((1 - TRAIN_VALIDATION_SPLIT_AT) * 100)))

c = 0

for i in list_of_cats_fnames:

    if c < (round(TRAIN_VALIDATION_SPLIT_AT * EXPERIMENT_SIZE)):

        copyfile(os.path.join(tmp_dir, i), os.path.join(train_cats_dir, i))

    else:

        copyfile(os.path.join(tmp_dir, i), os.path.join(validation_cats_dir, i))

    c += 1

    if c >= EXPERIMENT_SIZE:

        break



c = 0

for i in list_of_dogs_fnames:

    if c < (round(TRAIN_VALIDATION_SPLIT_AT * EXPERIMENT_SIZE)):

        copyfile(os.path.join(tmp_dir, i), os.path.join(train_dogs_dir, i))

    else:

        copyfile(os.path.join(tmp_dir, i), os.path.join(validation_dogs_dir, i))

    c += 1

    if c >= EXPERIMENT_SIZE:

        break



print('Total training cat images :', len(os.listdir(train_cats_dir)))

print('Total training dog images :', len(os.listdir(train_dogs_dir)))



print('Total validation cat images :', len(os.listdir(validation_cats_dir)))

print('Total validation dog images :', len(os.listdir(validation_dogs_dir)))

print('Loading images through generators ...')

# # Here we create ImageDataGenerator and we normalize while loading

train_datagen = ImageDataGenerator(

    rescale=1.0 / 255,

    rotation_range=15,

    shear_range=0.1,

    zoom_range=0.2,

    horizontal_flip=True,

    width_shift_range=0.1,

    height_shift_range=0.1

)

validation_datagen = ImageDataGenerator(

    rescale=1.0 / 255,

    rotation_range=15,

    shear_range=0.1,

    zoom_range=0.2,

    horizontal_flip=True,

    width_shift_range=0.1,

    height_shift_range=0.1)

print('Done')
# # # We then load data through the generator

train_generator = train_datagen.flow_from_directory(

    directory=train_dir,

    target_size=TARGET_SIZE,  # Resize the image while loading

    batch_size=BATCH_SIZE,  #

    class_mode='binary')  # 1 Dimensional binary labels, generator assigns 0 to cats, and 1 to dogs

# we can see that from train_generator.model.indicies



TOTAL_TRAINING = len(train_generator.filenames)

#

validation_generator = validation_datagen.flow_from_directory(

    directory=validation_dir,  # This is the source directory for training images

    target_size=TARGET_SIZE,  # All images will be resized to 150x150

    batch_size=BATCH_SIZE,

    class_mode='binary')



TOTAL_VALIDATION = len(validation_generator.filenames)
# I have started with simpler networks than this and with less number of samples and smaller number of training epochs,

# Now I am continuing from this network architecture Conve8MaxPool->Conv16MaxPool->Conv32MaxPool->Dense128

print('Constructing and compiling model ...')

model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=(128, 128, 3)),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')

])



model.compile(optimizer=RMSprop(lr=0.001),

              loss='binary_crossentropy',  # not sparse_crossentropy or categorical_corssentropy since

              # we are doing two class which can ben handled as a binary classification

              metrics=['accuracy'])

print('Done')
# here we train the model

print('Training ....')

history = model.fit(

    train_generator,

    epochs=NO_OF_EPOCHS,

    validation_data=validation_generator,

    steps_per_epoch=TOTAL_TRAINING / BATCH_SIZE,

    validation_steps=TOTAL_VALIDATION / BATCH_SIZE,

    callbacks = [callbacks],

    verbose=2)  # Found that this is the clearest, no annoying progress bars

print('Done')
# # -----------------------------------------------------------

# To have a healthy training, loss should decrease while accuracy increases

# if loss increases while accuracy increases then this is an overfitting case

acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']

#

epochs = range(len(acc))  # Get number of epochs

#

# # ------------------------------------------------

# # Plot training and validation accuracy per epoch

# # ------------------------------------------------

plt.plot(epochs, acc, color='b', label="Training accuracy")

plt.plot(epochs, val_acc, color='r', label="Validation accuracy")

plt.title('Training and validation accuracy')

plt.legend(loc='best', shadow=True)

plt.figure()

# #

# # ------------------------------------------------

# # Plot training and validation loss per epoch

# # ------------------------------------------------

plt.plot(epochs, loss, color='b', label="Training loss")

plt.plot(epochs, val_loss, color='r', label="Validation loss")

plt.title('Training and validation loss')

plt.legend(loc='best', shadow=True)

plt.show()