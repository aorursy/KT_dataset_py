# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os

import gc

import json

import math

import cv2

import PIL

from PIL import Image

import numpy as np



import matplotlib.pyplot as plt

import pandas as pd



import scipy

from tqdm import tqdm

%matplotlib inline

from keras.preprocessing import image
imSize = 128
train_df = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')

print(train_df.shape)

train_df.head()
def preprocess_image(image_path, desired_size=imSize):

    im = Image.open(image_path)

    im = im.resize((desired_size, )*2, resample=Image.LANCZOS)

    

    return im
# get the number of training images from the target\id dataset

N = train_df.shape[0]

# create an empty matrix for storing the images

x_train = np.empty((N, imSize, imSize, 3), dtype=np.uint8)



# loop through the images from the images ids from the target\id dataset

# then grab the cooresponding image from disk, pre-process, and store in matrix in memory

for i, image_id in enumerate(tqdm(train_df['image_name'])):

    x_train[i, :, :, :] = preprocess_image(

        f'../input/siim-isic-melanoma-classification/jpeg/train/{image_id}.jpg'

    )
np.save('x_train_128', x_train)