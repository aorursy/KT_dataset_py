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

from tensorflow.keras import layers

from tensorflow.keras.applications import DenseNet121

from tensorflow.keras.callbacks import Callback, ModelCheckpoint

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential

from tensorflow.keras.optimizers import Adam

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
# from kaggle_datasets import KaggleDatasets
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
AUTO = tf.data.experimental.AUTOTUNE

BATCH_SIZE = 36 * strategy.num_replicas_in_sync

EPOCHS = 20

# Data access

# GCS_DS_PATH = KaggleDatasets().get_gcs_path()
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

    im = np.array(im) / 255

    return im
%%time



# get the number of training images from the target\id dataset

# N = train_df.shape[0] # run on all data

N = 5650 # run on sample

# create an empty matrix for storing the images

x_train = np.empty((N, 224, 224, 3), dtype=np.float32)

# loop through the images from the images ids from the target\id dataset

# then grab the cooresponding image from disk, pre-process, and store in matrix in memory

for i, image_id in enumerate(tqdm(train_df['image_id'])):

    x_train[i, :, :, :] = preprocess_image(

        f'../input/prostate-cancer-grade-assessment/train_images/{image_id}.tiff'

    )

    # if sampling

    if i >= N-1:

        break
if os.path.exists(f'../input/prostate-cancer-grade-assessment/test_images'):

    # do the same thing as the last cell but on the test\holdout set

    N = test_df.shape[0]

    x_test = np.empty((N, 224, 224, 3), dtype=np.float32)

    for i, image_id in enumerate(tqdm(test_df['image_id'])):

        x_test[i, :, :, :] = preprocess_image(

            f'../input/prostate-cancer-grade-assessment/test_images/{image_id}.tiff'

        )
# pre-processing the target (i.e. one-hot encoding the target)

y_train = pd.get_dummies(train_df['isup_grade']).values.astype(np.int32)[0:N]



print(x_train.shape)

print(y_train.shape)

if os.path.exists('../input/prostate-cancer-grade-assessment/test_images'):

    print(x_test.shape)
%%time



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

    test_size=0.40, 

    random_state=2020

)
def convert(image, label):

    image = tf.image.convert_image_dtype(image, tf.float32) # Cast and normalize the image to [0,1]

    return image, label



def flip_aug(image,label):

    image,label = convert(image, label)

    image = tf.image.convert_image_dtype(image, tf.float32) # Cast and normalize the image to [0,1]

    image = tf.image.flip_left_right(image)

    return image,label



def rotate_aug(image,label):

    image,label = convert(image, label)

    image = tf.image.convert_image_dtype(image, tf.float32) # Cast and normalize the image to [0,1]

    image = tf.image.rot90(image)

    return image,label



def crop_aug(image,label):

    image,label = convert(image, label)

    image = tf.image.convert_image_dtype(image, tf.float32) # Cast and normalize the image to [0,1]

    image = tf.image.central_crop(image, central_fraction=0.8)

    return image,label





def pad_light_aug(image,label):

    image,label = convert(image, label)

    image = tf.image.convert_image_dtype(image, tf.float32) # Cast and normalize the image to [0,1]

    image = tf.image.resize_with_crop_or_pad(image, 230, 230) # Add 6 pixels of padding

    image = tf.image.random_crop(image, size=[224, 224, 3]) # Random crop back to 224x224x3

    image = tf.image.random_brightness(image, max_delta=0.5) # Random brightness

    return image,label
train_dataset = (

    tf.data.Dataset

    .from_tensor_slices((x_train, y_train))

    .repeat()

    .map(flip_aug, num_parallel_calls=AUTO)

    .map(rotate_aug, num_parallel_calls=AUTO)

    .map(crop_aug, num_parallel_calls=AUTO)

    .map(pad_light_aug, num_parallel_calls=AUTO)

    .batch(BATCH_SIZE)

    .prefetch(AUTO)

)



valid_dataset = (

    tf.data.Dataset

    .from_tensor_slices((x_val, y_val))

    .batch(BATCH_SIZE)

    .cache()

    .prefetch(AUTO)

)



if os.path.exists('../input/prostate-cancer-grade-assessment/test_images'):

    test_dataset = (

        tf.data.Dataset

        .from_tensor_slices(x_test)

        .batch(BATCH_SIZE)

    )
def build_model():

    densenet = DenseNet121(

    weights='../input/densenet-keras/DenseNet-BC-121-32-no-top.h5',

    include_top=False,

    input_shape=(224,224,3)

    )

    model = Sequential()

    model.add(densenet)

    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.Dropout(0.80))

    model.add(layers.Dense(6, activation='sigmoid'))

    return model
with strategy.scope():

    model = build_model()

    model.compile(

        loss='binary_crossentropy',

        optimizer=Adam(lr=0.00010509613402110064),

        metrics=['accuracy']

    )



model.summary()
n_steps = x_train.shape[0] // BATCH_SIZE



train_history = model.fit(

    train_dataset,

    steps_per_epoch=n_steps,

    validation_data=valid_dataset,

    epochs=EPOCHS

)
history_df = pd.DataFrame(train_history.history)

history_df[['loss', 'val_loss']].plot()

history_df[['accuracy', 'val_accuracy']].plot()
if os.path.exists('../input/prostate-cancer-grade-assessment/test_images'):

    y_test = model.predict(test_dataset)

    y_test = y_test > 0.37757874193797547

    y_test = y_test.astype(int).sum(axis=1) - 1

    test_df['isup_grade'] = y_test

    test_df.to_csv('submission.csv',index=False)

else: # if test is not available, just submit some random values

    rand_preds = []

    for i in range(len(test_df)):

        rand_preds.append(randint(0,5))

    test_df['isup_grade'] = rand_preds

    test_df = test_df[["image_id","isup_grade"]]

    test_df.to_csv('submission.csv',index=False)