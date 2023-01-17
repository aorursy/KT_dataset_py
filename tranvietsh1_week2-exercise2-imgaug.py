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
!pip3 install six numpy scipy Pillow matplotlib scikit-image opencv-python imageio Shapely

!pip install imgaug
import numpy as np 

import os

import keras

import matplotlib.pyplot as plt

import tensorflow as tf

import numpy as np

import imgaug as ia

import imgaug.augmenters as iaa

from keras import backend as K

from skimage import color, exposure, transform

import cv2

import glob
num_class = 2

BATCH_SIZE = 64

W = H = 128 #để size 32 mờ quá nên t để 128 cho nhìn rõ hơn

classes = ['cat', 'dog']
train_folder = "../input/dogandcat/dogandcat/train/"



classes = ["cat", "dog"]



data = []

label = []



def preprocessing(img):

    return cv2.resize(img, (W,H))



for i in range(len(classes)):

    class_name = classes[i]

    image_names = glob.glob(train_folder +class_name+'/'+'*.jpg')

    np.random.shuffle(image_names)

    #print(image_names)

    for img in image_names:

        image = cv2.imread(img)

        image = preprocessing(image)

        data.append(image)

        label.append(i)
fig=plt.figure(figsize=(10, 10))

columns = 3

rows = 3

for i in range(1, columns*rows +1):

    k = np.random.randint(0,len(data))

    fig.add_subplot(rows, columns, i)

    plt.imshow(data[k])

    plt.xticks([])

    plt.yticks([])

    plt.title(classes[int(label[k])])

plt.show()
seq = iaa.Sequential([

    iaa.Crop(percent=(0,0.5)),

    iaa.Fliplr(0.7),

    iaa.GaussianBlur(sigma=(0, 3.0)),

    iaa.Affine(rotate=(-45, 45),

    )

])

images_aug = seq.augment_images(data)

fig=plt.figure(figsize=(10, 10))

columns = 3

rows = 3

for i in range(1, columns*rows +1):

    k = np.random.randint(0,len(images_aug))

    fig.add_subplot(rows, columns, i)

    plt.imshow(images_aug[k])

    plt.xticks([])

    plt.yticks([])

    plt.title(classes[int(label[k])])

plt.show()