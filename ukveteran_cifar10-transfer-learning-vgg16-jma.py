# Pandas and Numpy for data structures and util fucntions

import scipy as sp

import numpy as np

import pandas as pd

from numpy.random import rand

pd.options.display.max_colwidth = 600



# Scikit Imports

from sklearn import preprocessing

from sklearn.metrics import roc_curve, auc, precision_recall_curve

from sklearn.model_selection import train_test_split





# Matplot Imports

import matplotlib.pyplot as plt

params = {'legend.fontsize': 'x-large',

          'figure.figsize': (15, 5),

          'axes.labelsize': 'x-large',

          'axes.titlesize':'x-large',

          'xtick.labelsize':'x-large',

          'ytick.labelsize':'x-large'}



plt.rcParams.update(params)

%matplotlib inline



# pandas display data frames as tables

from IPython.display import display, HTML



import warnings

warnings.filterwarnings('ignore')
import tensorflow as tf

from keras import callbacks

from keras import optimizers

from keras.datasets import cifar10

from keras.engine import Model

from keras.applications import vgg16 as vgg

from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D,BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from keras.utils import np_utils
BATCH_SIZE = 32

EPOCHS = 40

NUM_CLASSES = 10

LEARNING_RATE = 1e-4

MOMENTUM = 0.9
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train, X_val, y_train, y_val = train_test_split(X_train, 

                                                  y_train, 

                                                  test_size=0.15, 

                                                  stratify=np.array(y_train), 

                                                  random_state=42)
Y_train = np_utils.to_categorical(y_train, NUM_CLASSES)

Y_val = np_utils.to_categorical(y_val, NUM_CLASSES)

Y_test = np_utils.to_categorical(y_test, NUM_CLASSES)
base_model = vgg.VGG16(weights='imagenet', 

                       include_top=False, 

                       input_shape=(48, 48, 3))
# Extract the last layer from third block of vgg16 model

last = base_model.get_layer('block3_pool').output
# Add classification layers on top of it

x = GlobalAveragePooling2D()(last)

x= BatchNormalization()(x)

x = Dense(256, activation='relu')(x)

x = Dense(256, activation='relu')(x)

x = Dropout(0.6)(x)

pred = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(base_model.input, pred)
for layer in base_model.layers:

     layer.trainable = False
model.compile(loss='binary_crossentropy',

              optimizer=optimizers.Adam(lr=LEARNING_RATE),

              metrics=['accuracy'])
model.summary()
# prepare data augmentation configuration

train_datagen = ImageDataGenerator(

    rescale=1. / 255,

    horizontal_flip=False)
train_datagen.fit(X_train)

train_generator = train_datagen.flow(X_train,

                                     Y_train, 

                                     batch_size=BATCH_SIZE)
val_datagen = ImageDataGenerator(rescale=1. / 255,

    horizontal_flip=False)



val_datagen.fit(X_val)

val_generator = val_datagen.flow(X_val,

                                 Y_val,

                                 batch_size=BATCH_SIZE)
train_steps_per_epoch = X_train.shape[0] // BATCH_SIZE

val_steps_per_epoch = X_val.shape[0] // BATCH_SIZE



history = model.fit_generator(train_generator,

                              steps_per_epoch=train_steps_per_epoch,

                              validation_data=val_generator,

                              validation_steps=val_steps_per_epoch,

                              epochs=EPOCHS,

                              verbose=1)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

t = f.suptitle('Deep Neural Net Performance', fontsize=12)

f.subplots_adjust(top=0.85, wspace=0.3)



epochs = list(range(1,EPOCHS+1))

ax1.plot(epochs, history.history['accuracy'], label='Train Accuracy')

ax1.plot(epochs, history.history['val_accuracy'], label='Validation Accuracy')

ax1.set_xticks(epochs)

ax1.set_ylabel('Accuracy Value')

ax1.set_xlabel('Epoch')

ax1.set_title('Accuracy')

l1 = ax1.legend(loc="best")



ax2.plot(epochs, history.history['loss'], label='Train Loss')

ax2.plot(epochs, history.history['val_loss'], label='Validation Loss')

ax2.set_xticks(epochs)

ax2.set_ylabel('Loss Value')

ax2.set_xlabel('Epoch')

ax2.set_title('Loss')

l2 = ax2.legend(loc="best")