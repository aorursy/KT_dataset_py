# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import PIL

import cv2

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os
import matplotlib.pyplot as plt
train = pd.read_csv('/kaggle/input/kaggledays-china/train.csv')
train.shape
train.head()
star_pics = train[train['is_star']==1]
star_pics.head()
!ls /kaggle/input/kaggledays-china/train/train
img = cv2.imread('/kaggle/input/kaggledays-china/train3c/train3c/star/02dd9d6fa55c9d5aac64e001ab72dcd2.png')

print(img.shape)
star_pics_list = star_pics['id'].tolist()

star_x = star_pics['loc_x'].tolist()

star_y = star_pics['loc_y'].tolist()

plt.figure(figsize=(20,20))



for i in range(8):

    plt.subplot(1,8,i + 1)

    img = cv2.imread('/kaggle/input/kaggledays-china/train3c/train3c/star/%s.png'%star_pics_list[i])

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    

    plt.imshow(img)



# plt.figure(figsize=(20,20))

# for i in range(8):

#     plt.subplot(1,8,i + 1)

#     img = cv2.imread('/kaggle/input/kaggledays-china/train3c/train3c/star/%s.png'%star_pics_list[i])

#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     cv2.circle(img, (star_x[i], star_y[i]), 5, (255,0,0))

#     plt.imshow(img)



plt.figure(figsize=(20,20))



for i in range(8):

    plt.subplot(1,8,i + 1)

    img = cv2.imread('/kaggle/input/kaggledays-china/train/train/star/%s_a.png'%star_pics_list[i])

    cv2.circle(img, (star_x[i], star_y[i]), 5, (255,0,0))



    plt.imshow(img)

    

plt.figure(figsize=(20,20))



for i in range(8):

    plt.subplot(1,8,i + 1)

    img = cv2.imread('/kaggle/input/kaggledays-china/train/train/star/%s_b.png'%star_pics_list[i])

    cv2.circle(img, (star_x[i], star_y[i]), 5, (255,0,0))

    plt.imshow(img)

    

plt.figure(figsize=(20,20))

for i in range(8):

    plt.subplot(1,8,i + 1)

    img = cv2.imread('/kaggle/input/kaggledays-china/train/train/star/%s_c.png'%star_pics_list[i])

    cv2.circle(img, (star_x[i], star_y[i]), 5, (255,0,0))

    plt.imshow(img)
no_star_pics = train[train['is_star']==0]
no_star_pics.head()
nonstar_pics_list = no_star_pics['id'].tolist()

plt.figure(figsize=(20,20))



for i in range(8):

    plt.subplot(1,8,i + 1)

    img = cv2.imread('/kaggle/input/kaggledays-china/train3c/train3c/nonstar/%s.png'%nonstar_pics_list[i])

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img)

    

plt.figure(figsize=(20,20))

for i in range(8):

    plt.subplot(1,8,i + 1)

    img = cv2.imread('/kaggle/input/kaggledays-china/train/train/nonstar/%s_a.png'%nonstar_pics_list[i])

    plt.imshow(img)

    

plt.figure(figsize=(20,20))

for i in range(8):

    plt.subplot(1,8,i + 1)

    img = cv2.imread('/kaggle/input/kaggledays-china/train/train/nonstar/%s_b.png'%nonstar_pics_list[i])

    plt.imshow(img)

    

plt.figure(figsize=(20,20))

for i in range(8):

    plt.subplot(1,8,i + 1)

    img = cv2.imread('/kaggle/input/kaggledays-china/train/train/nonstar/%s_c.png'%nonstar_pics_list[i])

    plt.imshow(img)