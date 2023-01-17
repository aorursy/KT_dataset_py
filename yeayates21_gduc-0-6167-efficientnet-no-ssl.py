import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
from PIL import Image

from tqdm import tqdm

import glob

import gc

import matplotlib.pyplot as plt

%matplotlib inline
import json

import math

# import cv2

# import PIL

# from PIL import Image

# import numpy as np

import tensorflow as tf

from tensorflow.keras import layers

# from keras.applications import DenseNet121

from tensorflow.keras.callbacks import Callback, ModelCheckpoint

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential

from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt

# import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import cohen_kappa_score, accuracy_score

import scipy

from tqdm import tqdm

from sklearn.model_selection import train_test_split

from random import random

from sklearn import metrics
!pip install /kaggle/input/efficientnet-tfkeras-001/efficientnet-master/efficientnet-master
import efficientnet.tfkeras as efn
BATCH_SIZE = 15

TRAIN_VAL_RATIO = 0.10

EPOCHS = 4

LR = 0.00009409613402110064

imSize = 224
train_df = pd.read_csv("/kaggle/input/garage-detection-unofficial-ssl-challenge/image_labels_train.csv")

train_df.head()
Image.open("/kaggle/input/garage-detection-unofficial-ssl-challenge/GarageImages/GarageImages/image1607.jpg")
def preprocess_image(image_path, desired_size=224):

    im = Image.open(image_path)

    im = im.resize((desired_size, desired_size))

    im = np.array(im)

    if len(im.shape)==3:

        return im

    else:

        im2 = np.empty((desired_size, desired_size, 3), dtype=np.uint8)

        for j in range(3):

            im2[:,:,j] = im

        return im2
# get the number of training images from the target\id dataset

N = train_df.shape[0]

# create an empty matrix for storing the images

x_train = np.empty((N, imSize, imSize, 3), dtype=np.uint8)

# loop through the images from the images ids from the target\id dataset

# then grab the cooresponding image from disk, pre-process, and store in matrix in memory

for i, image_id in enumerate(tqdm(train_df['ID'])):

    x_train[i, :, :, :] = preprocess_image(

        f'/kaggle/input/garage-detection-unofficial-ssl-challenge/GarageImages/GarageImages/image{image_id}.jpg'

    )
holdout_df = pd.read_csv("/kaggle/input/garage-detection-unofficial-ssl-challenge/image_labels_holdout.csv")

holdout_df.head()
# do the same thing as the last cell but on the test\holdout set

N = holdout_df.shape[0]

x_holdout = np.empty((N, imSize, imSize, 3), dtype=np.uint8)

for i, image_id in enumerate(tqdm(holdout_df['ID'])):

    x_holdout[i, :, :, :] = preprocess_image(

        f'/kaggle/input/garage-detection-unofficial-ssl-challenge/GarageImages/GarageImages/image{image_id}.jpg'

    )
def get_auc(X,Y):

    probabilityOf1 = model.predict_proba(X)[:,1]

    fpr, tpr, thresholds = metrics.roc_curve(Y, probabilityOf1, pos_label=1)

    return metrics.auc(fpr, tpr)
x_train, x_val, y_train, y_val = train_test_split(

    x_train,  train_df['GarageDoorEntranceIndicator'],

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
effnet = efn.EfficientNetB3(weights=None,

                            input_shape=(imSize, imSize, 3),

                            include_top=False)

effnet.load_weights("../input/efficientnet-keras-weights-b0b5/efficientnet-b3_imagenet_1000_notop.h5")
model = tf.keras.Sequential()

model.add(effnet)

model.add(layers.GlobalAveragePooling2D())

model.add(layers.Dense(1, activation='sigmoid'))



model.compile(

    optimizer=Adam(lr=LR),

    loss = 'binary_crossentropy',

    metrics=[tf.keras.metrics.AUC()]

)

model.summary()
history = model.fit_generator(

    data_generator,

    steps_per_epoch=x_train.shape[0] // BATCH_SIZE,

    epochs=EPOCHS,

    validation_data=(x_val, y_val)

)
history_df = pd.DataFrame(history.history)

history_df[['loss', 'val_loss']].plot()

history_df[['auc', 'val_auc']].plot()
holdoutPreds = model.predict_proba(x_holdout)

fpr, tpr, thresholds = metrics.roc_curve(holdout_df['GarageDoorEntranceIndicator'], holdoutPreds, pos_label=1)

print("final holdout auc: ", metrics.auc(fpr, tpr))