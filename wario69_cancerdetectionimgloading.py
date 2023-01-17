# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.





"""Paths for various folders of images."""



CLLPath = "/kaggle/input/lymphoma/CLL"

MCLPath = "/kaggle/input/lymphoma/MCL"

FLPath = "/kaggle/input/lymphoma/FL"
import keras

import cv2

import matplotlib.pyplot as plt
"""Loading data from the files mentioned above"""

imgData = None

patchSize = 100

imageSize = 32

no_of_imgs = 10

batchSize = 50



def loadImages():

    """This method creates a DataFrame of images with their output"""

    global imgData

    imgs = []

    imgsOut = []

    imgs.extend(list(map(lambda x: CLLPath + "/" + x, os.listdir(CLLPath))))

    imgsOut.extend([0]*len(os.listdir(CLLPath)))

    imgs.extend(list(map(lambda x: MCLPath + "/" + x, os.listdir(MCLPath))))

    imgsOut.extend([1]*len(os.listdir(MCLPath)))

    imgs.extend(list(map(lambda x: FLPath + "/" + x, os.listdir(FLPath))))

    imgsOut.extend([2]*len(os.listdir(FLPath)))

    imgData = pd.DataFrame({"Output":imgsOut}, index = imgs).sample(frac = 1)

    del imgs, imgsOut



loadImages()

print(imgData.shape)
train_size = int(imgData.shape[0]*0.8)

train_img_data = imgData.iloc[:train_size, :]

test_img_data = imgData.iloc[train_size:,:]



print(train_img_data.shape)

print(test_img_data.shape)

    

def genPatches(image):

    global patchSize

    imgPatches = []

    imgDim = image.shape[0]

    for i in range(0, imgDim, patchSize):

        for j in range(0, imgDim, patchSize):

            imgPatch = image[i : i + patchSize, j : j + patchSize, :]

            imgPatch = cv2.resize(imgPatch, (32,32))

            imgPatches.append(imgPatch)

    return imgPatches, len(imgPatches)
def readImages(start = 0,end = -1,dim=(1500,1500),name='train'):

    

    x = []

    y = []

    data = 0

    

    if name == 'train':

        data  = train_img_data

    else:

        data = test_img_data

    

    end = data.shape[0] if end>data.shape[0] else end



    for i in range(start, end):

        img = cv2.imread(data.index[i],cv2.IMREAD_COLOR)

        img = cv2.resize(img,dim,interpolation=cv2.INTER_AREA)

        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        imgPatches, no_of_patches = genPatches(img)

        x.extend(imgPatches)

        y.extend([data.Output[i]]*no_of_patches)

    

    data = pd.DataFrame({"Images" : x, "OutPut" : y}).sample(frac = 1).reset_index(drop = True)

    

    if name == 'train':

        return data

    else:

        return data



df = readImages(end = no_of_imgs, name = 'test')

img = cv2.imread(imgData.index[1],cv2.IMREAD_COLOR)

img = cv2.resize(img,(1500,1500),interpolation=cv2.INTER_AREA)

img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)



plt.figure(figsize = (20,20))

plt.imshow(img)
img = img[0:150, 0:150, :]

plt.figure(figsize = (10,10))

plt.imshow(img)
plt.figure(figsize = (20,20))

plt.subplot(121)

img1 = cv2.resize(img, (50,50))

plt.imshow(img1)

plt.subplot(122)

img1 = cv2.resize(img, (32,32))

plt.imshow(img1)