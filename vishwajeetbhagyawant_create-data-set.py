# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from random import shuffle

import glob
shuffle_data = True  # shuffle the addresses



hdf5_path = '/kaggle/working/cats_train_data_set_1.hdf5'  # file path for the created .hdf5 file



cat_dog_train_path = '/kaggle/input/cat-and-dog/test_set/test_set/cats/*.jpg' # the original data path



# get all the image paths 

addrs = glob.glob(cat_dog_train_path)

# label the data as 0=cat, 1=dog

labels = np.ones(len(addrs))
# shuffle data

if shuffle_data:

    c = list(zip(addrs, labels)) # use zip() to bind the images and labels together

    shuffle(c)

 

    (addrs, labels) = zip(*c)  # *c is used to separate all the tuples in the list c,  

                               # "addrs" then contains all the shuffled paths and 

                               # "labels" contains all the shuffled labels.

                               

# Divide the data into 80% for train and 20% for test

train_addrs = addrs[0:len(addrs)]

train_labels = labels[0:len(labels)]
##################### second part: create the h5py object #####################

import numpy as np

import h5py



train_shape = (len(train_addrs), 500, 500, 3)



# open a hdf5 file and create earrays 

f = h5py.File(hdf5_path, mode='w')



# PIL.Image: the pixels range is 0-255,dtype is uint.

# matplotlib: the pixels range is 0-1,dtype is float.

f.create_dataset("train_img", train_shape, np.uint8)



# the ".create_dataset" object is like a dictionary, the "train_labels" is the key. 

f.create_dataset("train_labels", (len(train_addrs),), np.uint8)

f["train_labels"][...] = train_labels







######################## third part: write the images #########################

import cv2



# loop over train paths

for i in range(len(train_addrs)):

  

    if i % 1000 == 0 and i > 1:

        print ('Train data: {}/{}'.format(i, len(train_addrs)) )



    addr = train_addrs[i]

    img = cv2.imread(addr)

    img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_CUBIC)# resize to (128,128)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # cv2 load images as BGR, convert it to RGB

    f["train_img"][i, ...] = img[None] 





f.close()
train_set = h5py.File("/kaggle/working/cats_train_data_set_1.hdf5","r")
train_set.keys()
t = np.array(train_set['train_img'][:])
import matplotlib.pyplot as plt
t.shape
plt.imshow(t[100])