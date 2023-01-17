# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import json

import math

import cv2

import PIL

from PIL import Image

import numpy as np

from keras import layers

from keras.applications import DenseNet121

from keras.callbacks import Callback, ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.optimizers import Adam

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import cohen_kappa_score, accuracy_score

import scipy

import tensorflow as tf

from tqdm import tqdm

%matplotlib inline
import os

# There are two ways to load the data from the PANDA dataset:

# Option 1: Load images using openslide

import openslide

# Option 2: Load images using skimage (requires that tifffile is installed)

import skimage.io

# General packages

from IPython.display import display

# Plotly for the interactive viewer (see last section)

import plotly.graph_objs as go

# read images

import rasterio
import gc

from random import randint
# Detect hardware, return appropriate distribution strategy

try:

    # TPU detection. No parameters necessary if TPU_NAME environment variable is

    # set: this is always the case on Kaggle.

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.

    strategy = tf.distribute.get_strategy()



print("REPLICAS: ", strategy.num_replicas_in_sync)
BATCH_SIZE = 15

TRAIN_VAL_RATIO = 0.27

EPOCHS = 11

LR = 0.00010409613402110064
train_df = pd.read_csv('../input/prostate-cancer-grade-assessment/train.csv')

test_df = pd.read_csv('../input/prostate-cancer-grade-assessment/test.csv')

print(train_df.shape)

print(test_df.shape)

train_df.head()
test_df.head()
gc.collect()
def preprocess_image(image_path, desired_size=224):

    biopsy = openslide.OpenSlide(image_path)

    im = np.array(biopsy.get_thumbnail(size=(desired_size,desired_size)))

    im = Image.fromarray(im)

    im = im.resize((desired_size,desired_size)) 

    im = np.array(im)

    return im
# get the number of training images from the target\id dataset

N = train_df.shape[0]

# create an empty matrix for storing the images

x_train = np.empty((N, 224, 224, 3), dtype=np.uint8)

# loop through the images from the images ids from the target\id dataset

# then grab the cooresponding image from disk, pre-process, and store in matrix in memory

for i, image_id in enumerate(tqdm(train_df['image_id'])):

    x_train[i, :, :, :] = preprocess_image(

        f'../input/prostate-cancer-grade-assessment/train_images/{image_id}.tiff'

    )
if os.path.exists('../input/prostate-cancer-grade-assessment/test_images'):

    # do the same thing as the last cell but on the test\holdout set

    N = test_df.shape[0]

    x_test = np.empty((N, 224, 224, 3), dtype=np.uint8)

    for i, image_id in enumerate(tqdm(test_df['image_id'])):

        x_test[i, :, :, :] = preprocess_image(

            f'../input/prostate-cancer-grade-assessment/test_images/{image_id}.tiff'

        )

else:

    print("test images not found")
# pre-processing the target (i.e. one-hot encoding the target)

y_train = pd.get_dummies(train_df['isup_grade']).values



print(x_train.shape)

print(y_train.shape)

if os.path.exists('../input/prostate-cancer-grade-assessment/test_images'):

    print(x_test.shape)

else:

    print("test images not found")
# Further target pre-processing



# Instead of predicting a single label, we will change our target to be a multilabel problem; 

# i.e., if the target is a certain class, then it encompasses all the classes before it. 

# E.g. encoding a class 4 retinopathy would usually be [0, 0, 0, 1], 

# but in our case we will predict [1, 1, 1, 1]. For more details, 

# please check out Lex's kernel.



y_train_multi = np.empty(y_train.shape, dtype=y_train.dtype)

y_train_multi[:, 5] = y_train[:, 5]



for i in range(4, -1, -1):

    y_train_multi[:, i] = np.logical_or(y_train[:, i], y_train_multi[:, i+1])



print("Original y_train:", y_train.sum(axis=0))

print("Multilabel version:", y_train_multi.sum(axis=0))
x_train, x_val, y_train, y_val = train_test_split(

    x_train, y_train_multi, 

    test_size=TRAIN_VAL_RATIO, 

    random_state=2020

)
def create_datagen():

    return ImageDataGenerator(

        zoom_range=0.15,  # set range for random zoom

        # set mode for filling points outside the input boundaries

        fill_mode='constant',

        cval=0.,  # value used for fill_mode = "constant"

        horizontal_flip=True,  # randomly flip images

        vertical_flip=True,  # randomly flip images

    )



# Using original generator

data_generator = create_datagen().flow(x_train, y_train, batch_size=BATCH_SIZE, seed=2019)
densenet = DenseNet121(

    weights='../input/densenet-keras/DenseNet-BC-121-32-no-top.h5',

    include_top=False,

    input_shape=(224,224,3)

)
def build_model(LR=LR):

    model = Sequential()

    model.add(densenet)

    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.Dropout(0.80))

    model.add(layers.Dense(6, activation='sigmoid'))

    

    model.compile(

        loss='binary_crossentropy',

        optimizer=Adam(lr=LR),

        metrics=['accuracy']

    )

    

    return model
model = build_model()

model.summary()
history = model.fit_generator(

    data_generator,

    steps_per_epoch=x_train.shape[0] / BATCH_SIZE,

    epochs=EPOCHS,

    validation_data=(x_val, y_val)

)
history_df = pd.DataFrame(history.history)

history_df[['loss', 'val_loss']].plot()

history_df[['accuracy', 'val_accuracy']].plot()
if os.path.exists('../input/prostate-cancer-grade-assessment/test_images'):

    print("test images found.")

    y_test = model.predict(x_test)

    y_test = y_test > 0.37757874193797547

    y_test = y_test.astype(int).sum(axis=1) - 1

    test_df['isup_grade'] = y_test

    test_df = test_df[["image_id","isup_grade"]]

    test_df.to_csv('submission.csv',index=False)

else: # if test is not available, just submit some random values

    print("test images not found, submitting random values.")

    rand_preds = []

    for i in range(len(test_df)):

        rand_preds.append(randint(0,5))

    test_df['isup_grade'] = rand_preds

    test_df = test_df[["image_id","isup_grade"]]

    test_df.to_csv('submission.csv',index=False)