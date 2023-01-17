import os

import numpy as np

from tqdm import tqdm

from glob import glob

from PIL import Image

import cv2

from matplotlib import pylab as plt
img_size=500

num_train=len(glob('../input/building/facade/train_picture/'+'*.png'))

train_images=np.zeros((num_train,img_size,img_size,3),dtype=np.uint8)

train_labels=np.zeros((num_train,img_size,img_size,3),dtype=np.uint8)
cnt=0

for filename in tqdm(sorted(glob('../input/building/facade/train_picture/'+'*.png'))):

    img=Image.open(filename)

    img=img.resize((img_size,img_size),Image.LANCZOS) #图片resize

    arr=np.asarray(img)

    train_images[cnt,:,:,:]=arr

    cnt+=1
fig,axs=plt.subplots(2,2)

k=0

for i in range(2):

    for j in range(2):

        axs[i,j].imshow(train_images[k])

        axs[i,j].axis('off')

        k+=1
cnt=0

for filename in tqdm(sorted(glob('../input/building/facade/train_label/'+'*.jpg'))):

    img=Image.open(filename)

    img=img.resize((img_size,img_size),Image.LANCZOS)

    arr=np.asarray(img)

    train_labels[cnt,:,:,:]=arr

    cnt+=1
r,c=2,2

fig,axs=plt.subplots(r,c)

k=0

for i in range(r):

    for j in range(c):

        axs[i,j].imshow(train_labels[k])

        axs[i,j].axis('off')

        k+=1
print(train_images.shape)

print(train_labels.shape)
train_images=(train_images-127.5)/127.5