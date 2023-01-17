!wget --no-check-certificate https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 -O inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5
import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import zipfile

import matplotlib.pyplot as plt

from keras.callbacks import ReduceLROnPlateau

from tensorflow.keras import Model

from tensorflow.keras import layers

from tensorflow.keras.applications.inception_v3 import InceptionV3

from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.preprocessing.image import ImageDataGenerator

local_weights_file = 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(input_shape=(128, 128, 3),

                                include_top=False,

                                weights=None)
pre_trained_model.load_weights(local_weights_file)
for layer in pre_trained_model.layers:

    layer.trainable = False
last_layer = pre_trained_model.get_layer('mixed9')

print('last layer output shape: ', last_layer.output_shape)

last_output = last_layer.output
# Adding one output layer for our binary classification Cats vs Dogs task

# Flatten the output layer to 1 dimension

x = layers.Flatten()(last_output)

# Add a fully connected layer with 1,024 hidden units and ReLU activation

x = layers.Dense(1024, activation='relu')(x)

# Add a dropout rate of 0.2

x = layers.Dropout(0.2)(x)

# Add a final sigmoid layer for classification

x = layers.Dense(1, activation='sigmoid')(x)



# Build the new model

model = Model(pre_trained_model.input, x)



model.compile(optimizer=RMSprop(lr=0.0001),

              loss='binary_crossentropy',

              metrics=['accuracy'])
from os import makedirs

from shutil import rmtree

def make_directory(dir_path):

    if os.path.exists(dir_path):

        rmtree(dir_path)

    makedirs(dir_path)

    print(dir_path, ' folder is created')

    

base_dir = '../output/cats-vs-dogs'

tmp_dir = '../output/tmp'



make_directory(base_dir)

make_directory(tmp_dir)



# extract train data

with zipfile.ZipFile('../input/dogs-vs-cats/train.zip', 'r') as zip_ref:

    zip_ref.extractall(tmp_dir)

    zip_ref.close()
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',

                                            patience=2,

                                            verbose=1,

                                            factor=0.5,

                                            min_lr=0.00001)

callbacks = [learning_rate_reduction]

train_dir = os.path.join(base_dir, 'train')

validation_dir = os.path.join(base_dir, 'validation')



train_cats_dir = os.path.join(train_dir, 'cats')  # Directory with our training cat pictures

train_dogs_dir = os.path.join(train_dir, 'dogs')  # Directory with our training dog pictures

validation_cats_dir = os.path.join(validation_dir, 'cats')  # Directory with our validation cat pictures

validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # Directory with our validation dog pictures

print('Creating folders ....')

make_directory(train_dir)

make_directory(train_cats_dir)

make_directory(train_dogs_dir)

make_directory(validation_dir)

make_directory(validation_cats_dir)

make_directory(validation_dogs_dir)
list_of_fnames = os.listdir(os.path.join(tmp_dir,'train'))

print(len(list_of_fnames))

list_of_cats_fnames = [i for i in list_of_fnames if 'CAT' in i.upper()]

print('Found {0} CATS images in input folder tmp/train'.format(len(list_of_cats_fnames)))

list_of_dogs_fnames = [i for i in list_of_fnames if 'DOG' in i.upper()]

print('Found {0} DOGS images in input folder tmp/train'.format(len(list_of_dogs_fnames)))
np.random.shuffle(list_of_cats_fnames)

np.random.shuffle(list_of_dogs_fnames)
TOTAL_CATS = len(list_of_cats_fnames)

TOTAL_DOGS = len(list_of_dogs_fnames)
TRAIN_VALIDATION_SPLIT_AT = 0.6

BATCH_SIZE = 100

TARGET_SIZE = (128, 128)

NO_OF_EPOCHS = 5

EXPERIMENT_SIZE = 5000  # Here we are using only 5000 samples, we relying to Inception pretrained network and therefore

# we don't have to train on the full data set.
print('\nDistributing images to \n {0} \n {1} \n {2} \n {3}'

      '\nsuch that {4}% of total number of images goes to training and \n'

      '{5}% goes to validation'.format(

    train_cats_dir, train_dogs_dir,

    validation_cats_dir, validation_dogs_dir,

    round(TRAIN_VALIDATION_SPLIT_AT * 100),

    round((1 - TRAIN_VALIDATION_SPLIT_AT) * 100)))
# Copy images from tmp_dir to train/Cats, train/Dogs and to validation/Cats and validation/Dogs

# according to the split percentage we decided.



from shutil import copyfile

tmp_train_dir = os.path.join(tmp_dir, 'train')

c = 0

for i in list_of_cats_fnames:

    if c < (round(TRAIN_VALIDATION_SPLIT_AT * EXPERIMENT_SIZE)):

        copyfile(os.path.join(tmp_train_dir, i), os.path.join(train_cats_dir, i))

    else:

        copyfile(os.path.join(tmp_train_dir, i), os.path.join(validation_cats_dir, i))

    c += 1

    if c >= EXPERIMENT_SIZE:

        break



c = 0

for i in list_of_dogs_fnames:

    if c < (round(TRAIN_VALIDATION_SPLIT_AT * EXPERIMENT_SIZE)):

        copyfile(os.path.join(tmp_train_dir, i), os.path.join(train_dogs_dir, i))

    else:

        copyfile(os.path.join(tmp_train_dir, i), os.path.join(validation_dogs_dir, i))

    c += 1

    if c >= EXPERIMENT_SIZE:

        break



print('Total training cat images :', len(os.listdir(train_cats_dir)))

print('Total training dog images :', len(os.listdir(train_dogs_dir)))



print('Total validation cat images :', len(os.listdir(validation_cats_dir)))

print('Total validation dog images :', len(os.listdir(validation_dogs_dir)))
print('Loading images through generators ...')

train_datagen = ImageDataGenerator(rescale=1. / 255.,

                                   rotation_range=40,

                                   width_shift_range=0.2,

                                   height_shift_range=0.2,

                                   shear_range=0.2,

                                   zoom_range=0.2,

                                   horizontal_flip=True)



# Note that the validation data should not be augmented!

validation_datagen = ImageDataGenerator(rescale=1.0 / 255.)



# Flow training images in batches of 20 using train_datagen generator

train_generator = train_datagen.flow_from_directory(train_dir,

                                                    batch_size=BATCH_SIZE,

                                                    target_size=TARGET_SIZE,

                                                    class_mode='binary')



TOTAL_TRAINING = len(train_generator.filenames)



validation_generator = validation_datagen.flow_from_directory(validation_dir,

                                                              batch_size=BATCH_SIZE,

                                                              target_size=TARGET_SIZE,

                                                              class_mode='binary')



TOTAL_VALIDATION = len(validation_generator.filenames)
history = model.fit(

    train_generator,

    epochs=NO_OF_EPOCHS,

    validation_data=validation_generator,

    steps_per_epoch=TOTAL_TRAINING / BATCH_SIZE,

    validation_steps=TOTAL_VALIDATION / BATCH_SIZE,

    callbacks=[callbacks],

    verbose=2)  # Found that this is the clearest, no annoying progress bars



print('Done')
# # -----------------------------------------------------------

# To have a health training Loss should decrease while accuracy increases

# if loss increase while accuracy increases then this is an overfitting case

acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']

#

epochs = range(len(acc))  # Get number of epochs

# # # ------------------------------------------------

# # # Plot training and validation accuracy per epoch

# # # ------------------------------------------------

plt.plot(epochs, acc, color='b', label="Training accuracy")

plt.plot(epochs, val_acc, color='r', label="Validation accuracy")

plt.title('Training and validation accuracy')

plt.legend(loc='best', shadow=True)

plt.show()

# ------------------------------------------------

# Plot training and validation loss per epoch

# ------------------------------------------------

plt.plot(epochs, loss, color='b', label="Training loss")

plt.plot(epochs, val_loss, color='r', label="Validation loss")

plt.title('Training and validation loss')

plt.legend(loc='best', shadow=True)

plt.show()
