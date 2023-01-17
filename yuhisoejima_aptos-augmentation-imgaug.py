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
import json

import math

import os



import cv2

from PIL import Image

import numpy as np

from keras import layers

from keras.applications import DenseNet121

from keras.callbacks import Callback, ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.optimizers import Adam

from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import cohen_kappa_score, accuracy_score

import scipy

from tqdm import tqdm

import imgaug as ia

from imgaug import augmenters as iaa



%matplotlib inline
train_df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')



print(train_df.shape)



train_df.head()
def display_samples(df, columns=5, rows=4):

    fig = plt.figure(figsize=(5*columns, 4*rows))

    

    for i in range(columns*rows):

        image_path = df.loc[i, 'id_code']

        image_id = df.loc[i, 'diagnosis']

        img = cv2.imread(f'../input/aptos2019-blindness-detection/train_images/{image_path}.png')

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        fig.add_subplot(rows, columns, i+1)

        plt.title(image_id)

        plt.imshow(img)

        

display_samples(train_df)
def crop_image_from_gray(img, tol=7):

    if img.ndim == 2:

        mask = img > tol

        return img[np.ix_(mask.any(1), mask.any(0))]

    

    elif img.ndim == 3:

        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        

        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]

        

        if check_shape == 0:

            return img

        else:

            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]

            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]

            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]

            img = np.stack([img1, img2, img3], axis=-1)

        return img
def circle_crop(img, sigmaX=10):   

    """

    Create circular crop around image centre    

    """    

    img = crop_image_from_gray(img)    

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    

    height, width, depth = img.shape    

    

    x = int(width/2)

    y = int(height/2)

    r = np.amin((x,y))

    



    circle_img = np.zeros((height, width), np.uint8)

    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)

    img = cv2.bitwise_and(img, img, mask=circle_img)

    img = crop_image_from_gray(img)

    img=cv2.addWeighted (img, 4, cv2.GaussianBlur(img, (0,0), sigmaX), -4, 128)

    

    return img 
train_num = train_df[:20].shape[0]

X_train = np.empty((train_num, 224, 224, 3), dtype=np.uint8)



for num, code in enumerate(tqdm(train_df[:20]['id_code'])):

    img = cv2.imread('../input/aptos2019-blindness-detection/train_images/' + code + '.png')

    img = circle_crop(img)

    img = cv2.resize(img, (224, 224))

    X_train[num, :, :, :] = img
columns=5

rows=4



fig = plt.figure(figsize=(5*columns, 4*rows))



for i in range(columns*rows):

    img = X_train[i]

    fig.add_subplot(rows, columns, i+1)

    plt.imshow(img)
def img_aug(X, columns=5, rows=4):

    aug_num = X.shape[0]

    

    #Augmentation

    seq = iaa.SomeOf(3, [iaa.Affine(rotate=(-10, 10)), #回転

                          iaa.Fliplr(0.5), #左右反転

                          iaa.Flipud(0.5), #上下反転

                          iaa.Affine(scale=(0.9, 1.1)), #拡大縮小

                          iaa.Multiply((0.7, 1.3)), #明暗

                          iaa.Dropout(0.02), #Dropout

                          iaa.ContrastNormalization((0.85, 1.15), per_channel=0.5)

                         ])

    images = [X[i] for i in range(aug_num)]

    image_aug = seq.augment_images(images)



    #show

    fig = plt.figure(figsize=(5*columns, 4*rows))

    for num, img in enumerate(image_aug):

        fig.add_subplot(rows, columns, num+1)

        plt.imshow(img)

        
img_aug(X_train)