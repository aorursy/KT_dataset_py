# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os

import sys

import random

import warnings



import numpy as np

import pandas as pd



import matplotlib.pyplot as plt



from tqdm import tqdm

from itertools import chain

from skimage.io import imread, imshow, imread_collection, concatenate_images

from skimage.transform import resize

from skimage.morphology import label



from keras.models import Model, load_model

from keras.layers import Input

from keras.layers.core import Dropout, Lambda

from keras.layers.convolutional import Conv2D, Conv2DTranspose

from keras.layers.pooling import MaxPooling2D

from keras.layers.merge import concatenate

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras import backend as K



import tensorflow as tf



%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
IMG_WIDTH = 128

IMG_HEIGHT = 128

IMG_CHANNELS = 3

TRAIN_PATH = '../input/stage1_train/'

TEST_PATH = '../input/stage1_test/'



warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

seed = 42

random.seed = seed

np.random.seed = seed
train_ids = next(os.walk(TRAIN_PATH))[1]

test_ids = next(os.walk(TEST_PATH))[1]
X_train = np.zeros((len(train_ids), IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS), dtype = np.uint8)

Y_train = np.zeros((len(train_ids), IMG_WIDTH, IMG_HEIGHT, 1), dtype = np.bool)

print("Loading training images and masks...")



for n, im_id in tqdm(enumerate(train_ids), total = len(train_ids)):

    path = TRAIN_PATH + im_id

    img = imread(path+'/images/'+im_id+'.png')[:,:,:IMG_CHANNELS]

    img = resize(img, (IMG_WIDTH, IMG_HEIGHT), mode = 'constant', preserve_range = True)

    X_train[n] = img

    mask = np.zeros((IMG_WIDTH, IMG_HEIGHT, 1))

    for mask_file in next(os.walk(path+'/masks/'))[2]:

        mask_ = imread(path + '/masks/' + mask_file)

        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True), axis=-1)

        mask = np.maximum(mask, mask_)

    Y_train[n] = mask



X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

sizes_test = []

print('Loading test images ... ')

for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):

    path = TEST_PATH + id_

    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]

    sizes_test.append([img.shape[0], img.shape[1]])

    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

    X_test[n] = img



print('Done!')
ix = random.randint(0, len(train_ids))

plt.imshow(X_train[ix])

plt.show()

plt.imshow(np.squeeze(Y_train[ix]), cmap='gray')
def mean_iou(y_true, y_pred):

    prec = []

    for t in np.arange(0.5, 1.0, 0.05):

        y_pred_ = tf.to_int32(y_pred > t)

        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)

        K.get_session().run(tf.local_variables_initializer())

        with tf.control_dependencies([up_opt]):

            score = tf.identity(score)

        prec.append(score)

    return K.mean(K.stack(prec), axis=0)
inputs = Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))

s = Lambda(lambda x: x/255)(inputs)



conv_1 = Conv2D(16, (3,3), activation="relu", padding = 'same')(s)

conv_1 = Dropout(0.1)(conv_1)



conv_1 = Conv2D(16, (3,3), activation="relu", padding = 'same')(conv_1)

pool_1 = MaxPooling2D((2, 2))(conv_1)



conv_2 = Conv2D(32, (3,3), activation="relu", padding = 'same')(pool_1)

conv_2= Dropout(0.1)(conv_2)



conv_2 = Conv2D(32, (3,3), activation="relu", padding = 'same')(conv_2)

pool_2 = MaxPooling2D((2, 2))(conv_2)



conv_3 = Conv2D(64, (3,3), activation="relu", padding = 'same')(pool_2)

conv_3 = Dropout(0.1)(conv_3)



conv_3 = Conv2D(64, (3,3), activation="relu", padding = 'same')(conv_3)

pool_3 = MaxPooling2D((2, 2))(conv_3)



conv_4 = Conv2D(128, (3,3), activation="relu", padding = 'same')(pool_3)

conv_4 = Dropout(0.1)(conv_4)



conv_4 = Conv2D(128, (3,3), activation="relu", padding = 'same')(conv_4)

pool_4 = MaxPooling2D((2, 2))(conv_4)



conv_5 = Conv2D(256, (3,3), activation="relu", padding = 'same')(pool_4)

conv_5 = Dropout(0.1)(conv_5)



conv_5 = Conv2D(256, (3,3), activation="relu", padding = 'same')(conv_5)

pool_5 = MaxPooling2D((2, 2))(conv_5)



ct_6 = Conv2DTranspose(128, (2,2), strides = (2,2), padding ='same')(conv_5)

ct_6 = concatenate([ct_6, conv_4])

conv_6 = Conv2D(128, (3,3), activation="relu", padding = 'same')(ct_6)

conv_6 = Dropout(0.1)(conv_6)

conv_6 = Conv2D(128, (3,3), activation="relu", padding = 'same')(conv_6)



ct_7 = Conv2DTranspose(64, (2,2), strides = (2,2), padding ='same')(conv_6)

ct_7 = concatenate([ct_7, conv_3])

conv_7 = Conv2D(64, (3,3), activation="relu", padding = 'same')(ct_7)

conv_7 = Dropout(0.1)(conv_7)

conv_7 = Conv2D(64, (3,3), activation="relu", padding = 'same')(conv_7)



ct_8 = Conv2DTranspose(32, (2,2), strides = (2,2), padding ='same')(conv_7)

ct_8 = concatenate([ct_8, conv_2])

conv_8 = Conv2D(32, (3,3), activation="relu", padding = 'same')(ct_8)

conv_8 = Dropout(0.1)(conv_8)

conv_8 = Conv2D(32, (3,3), activation="relu", padding = 'same')(conv_8)



ct_9 = Conv2DTranspose(16, (2,2), strides = (2,2), padding ='same')(conv_8)

ct_9 = concatenate([ct_9, conv_1])

conv_9 = Conv2D(16, (3,3), activation="relu", padding = 'same')(ct_9)

conv_9 = Dropout(0.1)(conv_9)

conv_9 = Conv2D(16, (3,3), activation="relu", padding = 'same')(conv_9)



outputs = Conv2D(1,(1,1), activation = 'sigmoid')(conv_9)



model = Model(inputs=[inputs], outputs=[outputs])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])

model.summary()
checkpoint = ModelCheckpoint('model-dsbowl2018-1.h5', verbose=1, save_best_only=True)

results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=50, 

                    callbacks=[checkpoint])
model = load_model('model-dsbowl2018-1.h5', custom_objects={'mean_iou': mean_iou})

pred_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)

pred_test = model.predict(X_test, verbose = 1)
pred_val_t = (pred_val > 0.5).astype(np.uint8)

pred_test_t = (pred_test > 0.5).astype(np.uint8)
ix = random.randint(0, len(preds_val))

plt.imshow(X_train[ix])

plt.show()

plt.imshow(np.squeeze(Y_train[ix]))

plt.show()

plt.imshow(np.squeeze(pred_val_t[ix]))