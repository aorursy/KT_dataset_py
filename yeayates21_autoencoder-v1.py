# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Import some packages to use

import cv2

import numpy as np

import pandas as pd

# matplotlib

import matplotlib.pyplot as plt

%matplotlib inline 

#To see our directory

import os

import random

import gc   #Gabage collector for cleaning deleted data from memory

# sklearn

from sklearn.model_selection import train_test_split

# keras

from keras import layers

from keras import models

from keras import optimizers

from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing.image import img_to_array, load_img

from keras.models import Sequential

from keras.layers import Flatten, Dropout, MaxPooling2D

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D

from keras.models import Model

from keras import backend as K

from keras.utils import to_categorical

from keras.callbacks import EarlyStopping, LearningRateScheduler

from keras.applications import DenseNet121

# math

import math
train_df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

ids_and_target = train_df[['id_code','diagnosis']]

onehottarget = pd.get_dummies(ids_and_target['diagnosis'])

ids_and_target = ids_and_target.join(onehottarget)

print(ids_and_target.head())
#densenet = DenseNet121(

#    weights='../input/densenet-keras/DenseNet-BC-121-32-no-top.h5',

#    include_top=False,

#    input_shape=(280,280,3)

#)
input_img = Input(shape=(280, 280, 3))  # adapt this if using `channels_first` image data format



#x = densenet(input_img)



x = Conv2D(40, kernel_size=3, activation='tanh', padding='same')(input_img)

x = MaxPooling2D(pool_size=2, padding='same')(x)

x = Conv2D(20, kernel_size=3, activation='tanh', padding='same')(x)

x = MaxPooling2D(pool_size=2, padding='same')(x)

x = Conv2D(8, kernel_size=3, activation='tanh', padding='same')(x)

encoded = MaxPooling2D(pool_size=2, padding='same')(x)



x = Conv2D(8, kernel_size=3, activation='tanh', padding='same')(encoded)

x = UpSampling2D(size=2)(x)

x = Conv2D(20, kernel_size=3, activation='tanh', padding='same')(x)

x = UpSampling2D(size=2)(x)

x = Conv2D(40, kernel_size=3, activation='tanh', padding='same')(x)

x = UpSampling2D(size=2)(x)

decoded = Conv2D(3, kernel_size=3, activation='sigmoid', padding='same')(x)
# this model maps an input to it's encoded represenations

encoder = Model(input_img, encoded)
# full autoencoder model

autoencoder = Model(input_img, decoded)
# create a placeholder for an encoded input

encoded_input = Input(shape=(35,35,8))

# retrieve the last layer of the autoencoder model

decoder_layer = autoencoder.layers[-6]

# create the decoder model

decoder = Model(encoded_input, decoder_layer(encoded_input))


autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.summary()
train_dir = '../input/aptos2019-blindness-detection/train_images'

test_dir = '../input/aptos2019-blindness-detection/test_images'



train_imgs = ['../input/aptos2019-blindness-detection/train_images/{}'.format(i) for i in os.listdir(train_dir)]  #get images

test_imgs = ['../input/aptos2019-blindness-detection/test_images/{}'.format(i) for i in os.listdir(test_dir)] #get test images



X_train, X_val = train_test_split(train_imgs, test_size=0.20, random_state=2)
def LoadImagesAndTarget(files, lookup_table, shape):

    # Initialize variables

    i = 0

    w = shape[0]

    h = shape[1]

    batch_images = np.zeros(((len(files)),w,h,3))

    #print("batch image shape:",batch_images.shape)

    targetvals = np.zeros(((len(files)),5))

    #print("target val shape: ",targetvals.shape)

    for file in files:

        # read in image

        img = cv2.imread(file)

        # resize

        newimage = cv2.resize(img, shape) / 255

        # add image to training set

        batch_images[i] = newimage

        #print("batch_images[i]",batch_images[i])

        # get the filename without extension

        filename = os.path.splitext(os.path.basename(file))[0]

        #print("filename: ",filename)

        # only keep the row from the lookup table that matches our id

        getrow = lookup_table.loc[lookup_table['id_code'] == filename]

        # change the format to one-hot encoded, and save to target dataset

        targetvals[i] = np.asmatrix(getrow[[0,1,2,3,4]].values[0], dtype=np.float32)

        # get target based on filename

        i += 1

    #print("returning batches ...")

    #print("batch image shape:",batch_images.shape)

    return batch_images, batch_images
def image_generator(IMGDATA, batchsize, lookup_table, shape):

    # needed for generator to work

    while True:

        # loop over all batches

        for i in range(math.ceil(len(IMGDATA)/batchsize)):

            # find the start and end index of the batch

            window_slide = batchsize*i

            start = min(0+window_slide,len(IMGDATA))

            end = min(batchsize+window_slide,len(IMGDATA))

            #print("Start: ",start)

            #print("End: ",end)

            # get images

            Xtrain, Xtrain = LoadImagesAndTarget(IMGDATA[start:end], lookup_table, shape)

            #print("batch ",i," completed!")

            yield [Xtrain, Xtrain]
def image_generator_val(IMGDATA, batchsize, lookup_table, shape):

    # needed for generator to work

    while True:

        # loop over all batches

        for i in range(math.ceil(len(IMGDATA)/batchsize)):

            # find the start and end index of the batch

            window_slide = batchsize*i

            start = min(0+window_slide,len(IMGDATA))

            end = min(batchsize+window_slide,len(IMGDATA))

            # get images

            Xval, Xval = LoadImagesAndTarget(IMGDATA[start:end], lookup_table, shape)

            yield [Xval, Xval]
nb_epoch = 50

batch_size = 32

n_steps = (len(X_train)) // batch_size

autoencoder.fit_generator(image_generator(X_train, batch_size, ids_and_target, (280, 280)),

                    steps_per_epoch=n_steps, 

                    epochs=nb_epoch, 

                    validation_data=image_generator_val(X_val, batch_size, ids_and_target, (280, 280)),

                    validation_steps=(len(X_val) // batch_size),

                    callbacks=[LearningRateScheduler(lambda _: 1.95)])
files = os.listdir('../input/aptos2019-blindness-detection/train_images')

i = 0

for file in files[0:2]:

    batch_images = np.zeros((1,280,280,3))

    # read in image

    img = cv2.imread('../input/aptos2019-blindness-detection/train_images/' + file)

    # resize

    imgp = cv2.resize(img, (280,280))

    # normalize

    imgp = np.array(imgp) / 255.0

    # reformat img

    batch_images[0] = imgp

    # score image

    decoded_img = autoencoder.predict(batch_images, verbose=0)

    # show image

    #f, axarr = plt.subplots(1,2)

    #axarr[0].imshow(img)

    #axarr[1].imshow(np.squeeze(decoded_img, axis=0))

    fig = plt.figure()

    a=fig.add_subplot(1,2,1)

    imgplot = plt.imshow(imgp)

    a.set_title('Original')

    a=fig.add_subplot(1,2,2)

    imgplot = plt.imshow(np.squeeze(decoded_img, axis=0))

    a.set_title('Decoded')
files = os.listdir('../input/aptos2019-blindness-detection/train_images')

i = 0

for file in files[0:2]:

    batch_images = np.zeros((1,280,280,3))

    # read in image

    img = cv2.imread('../input/aptos2019-blindness-detection/train_images/' + file)

    # resize

    imgp = cv2.resize(img, (280,280))

    # normalize

    imgp = np.array(imgp) / 255.0

    # reformat img

    batch_images[0] = imgp

    # score image

    decoded_img = autoencoder.predict(batch_images, verbose=0)

    # show image

    #f, axarr = plt.subplots(1,2)

    #axarr[0].imshow(img)

    #axarr[1].imshow(np.squeeze(decoded_img, axis=0))

    #fig = plt.figure()

    #a=fig.add_subplot(1,2,1)

    #imgplot = plt.imshow(imgp)

    #a.set_title('Original')

    #a=fig.add_subplot(1,2,2)

    #imgplot = plt.imshow(np.squeeze(decoded_img, axis=0))

    #a.set_title('Decoded')

    print("Encoded Image:")

    print(encoder.predict(batch_images, verbose=0).shape)

    print(encoder.predict(batch_images, verbose=0).flatten())

    print(encoder.predict(batch_images, verbose=0).flatten().shape)