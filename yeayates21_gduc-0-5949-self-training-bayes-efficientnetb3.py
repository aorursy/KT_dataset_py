# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

myStop = 0

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        myStop += 1

        print(os.path.join(dirname, filename))

        if myStop==20:

            break

        

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from PIL import Image

from tqdm import tqdm

import glob

import gc

import matplotlib.pyplot as plt

%matplotlib inline
from sklearn.naive_bayes import GaussianNB

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from random import random

from sklearn import metrics
train_df = pd.read_csv("/kaggle/input/garage-detection-unofficial-ssl-challenge/image_labels_train.csv")

train_df.head()
Image.open("/kaggle/input/garage-detection-unofficial-ssl-challenge/GarageImages/GarageImages/image1607.jpg")
np.array(Image.open("/kaggle/input/garage-detection-unofficial-ssl-challenge/GarageImages/GarageImages/image1607.jpg")).shape
np.array(Image.open("/kaggle/input/garage-detection-unofficial-ssl-challenge/GarageImages/GarageImages/image1607.jpg").resize((224, 224)))[:,:,0].flatten().shape
(224, )*2
224*224
def preprocess_image(image_path, desired_size=224):

    im = Image.open(image_path)

    im = im.resize((desired_size, )*2, resample=Image.LANCZOS)

    im = np.array(im)

    if len(im.shape)==3:

        im = im[:,:,0]

    im = im.flatten()

    return im
# get the number of training images from the target\id dataset

N = train_df.shape[0]

# create an empty matrix for storing the images

x_train = np.empty((N, 50176), dtype=np.uint8)



# loop through the images from the images ids from the target\id dataset

# then grab the cooresponding image from disk, pre-process, and store in matrix in memory

for i, image_id in enumerate(tqdm(train_df['ID'])):

    x_train[i, :] = preprocess_image(

        f'/kaggle/input/garage-detection-unofficial-ssl-challenge/GarageImages/GarageImages/image{image_id}.jpg'

    )
holdout_df = pd.read_csv("/kaggle/input/garage-detection-unofficial-ssl-challenge/image_labels_holdout.csv")

holdout_df.head()
# get the number of training images from the target\id dataset

N = holdout_df.shape[0]

# create an empty matrix for storing the images

x_holdout = np.empty((N, 50176), dtype=np.uint8)



# loop through the images from the images ids from the target\id dataset

# then grab the cooresponding image from disk, pre-process, and store in matrix in memory

for i, image_id in enumerate(tqdm(holdout_df['ID'])):

    x_holdout[i, :] = preprocess_image(

        f'/kaggle/input/garage-detection-unofficial-ssl-challenge/GarageImages/GarageImages/image{image_id}.jpg'

    )
unlabeledIDs = []

labeledIDs = holdout_df['ID'].tolist() + train_df['ID'].tolist()

for file in tqdm(glob.glob('/kaggle/input/garage-detection-unofficial-ssl-challenge/GarageImages/GarageImages/*.jpg')):

    myStart = file.find('/image')

    myEnd = file.find('.jpg')

    myID = file[myStart+6:myEnd]

    if int(myID) not in labeledIDs:

        unlabeledIDs.append(myID)
# get the number of training images from the target\id dataset

N = len(unlabeledIDs)

# create an empty matrix for storing the images

x_unlabeled = np.empty((N, 50176), dtype=np.uint8)



# loop through the images from the images ids from the target\id dataset

# then grab the cooresponding image from disk, pre-process, and store in matrix in memory

for i, image_id in enumerate(tqdm(unlabeledIDs)):

    x_unlabeled[i, :] = preprocess_image(

        f'/kaggle/input/garage-detection-unofficial-ssl-challenge/GarageImages/GarageImages/image{image_id}.jpg'

    )
def get_auc(X,Y):

    probabilityOf1 = model.predict_proba(X)[:,1]

    fpr, tpr, thresholds = metrics.roc_curve(Y, probabilityOf1, pos_label=1)

    return metrics.auc(fpr, tpr)
sslRounds = 4

x_train_ssl = np.concatenate((x_train, x_unlabeled), axis=0)

for sslRound in tqdm(range(sslRounds)):

    # define model

    model = MultinomialNB()

    # fit model

    if sslRound==0:

        # first round, fit on just labeled data

        model.fit(x_train, train_df['GarageDoorEntranceIndicator'])

    else:

        # all other rounds, fit on all data

        model.fit(x_train_ssl, y_train_ssl)

    # score unlabeled data

    predictions = model.predict_proba(x_unlabeled)[:,1]

    # set random threshold

    threshold = 0.5

    # print("threshold selected: ", threshold)

    # create pseudo lables based on threshold

    pseudoLabels = np.where(predictions>threshold,1,0)

    # add pseudo labels to next round of training 

    y_train_ssl = np.concatenate((train_df['GarageDoorEntranceIndicator'], pseudoLabels), axis=0)

    # clean up

    if sslRound<(sslRounds-1):

        del model

        gc.collect()
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

%matplotlib inline
!pip install /kaggle/input/efficientnet-tfkeras-001/efficientnet-master/efficientnet-master
import efficientnet.tfkeras as efn
BATCH_SIZE = 15

TRAIN_VAL_RATIO = 0.10

EPOCHS = 6

LR = 0.00010409613402110064

imSize = 224
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
NonHoldoutIDs = train_df['ID'].tolist() + unlabeledIDs

len(NonHoldoutIDs)
# get the number of training images from the target\id dataset

N = len(NonHoldoutIDs)

# create an empty matrix for storing the images

x_train = np.empty((N, imSize, imSize, 3), dtype=np.uint8)

# loop through the images from the images ids from the target\id dataset

# then grab the cooresponding image from disk, pre-process, and store in matrix in memory

for i, image_id in enumerate(tqdm(NonHoldoutIDs)):

    x_train[i, :, :, :] = preprocess_image(

        f'/kaggle/input/garage-detection-unofficial-ssl-challenge/GarageImages/GarageImages/image{image_id}.jpg'

    )
# do the same thing as the last cell but on the test\holdout set

N = holdout_df.shape[0]

x_holdout = np.empty((N, imSize, imSize, 3), dtype=np.uint8)

for i, image_id in enumerate(tqdm(holdout_df['ID'])):

    x_holdout[i, :, :, :] = preprocess_image(

        f'/kaggle/input/garage-detection-unofficial-ssl-challenge/GarageImages/GarageImages/image{image_id}.jpg'

    )
x_train, x_val, y_train, y_val = train_test_split(

    x_train, y_train_ssl,

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