# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import cv2
from skimage.data import imread
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_train_gt = pd.read_csv('../input/train_ship_segmentations.csv')
df_train_gt[(df_train_gt.EncodedPixels.isna() == False)].head()
pixels = df_train_gt[df_train_gt.ImageId == '000155de5.jpg'].EncodedPixels
img = cv2.imread('../input/train/' + '000155de5.jpg')
img_ = img.copy()
plt.figure()
plt.imshow(img)
def show_segmentation(img, pixels):
    img_seg = np.zeros(img.shape[:-1])
    for pixel_ in pixels:
        pixels_ = pixel_.split()
        pixels_ = np.array(pixels_ ,dtype=int)
        for i in range(0, len(pixels_), 2):
            y = int(pixels_[i]/img.shape[0])
            x = int(pixels_[i]%img.shape[0])
            step = int(pixels_[i+1])
            img[x:x+step,y] = 255
            img_seg[x:x+step,y] = 255
    plt.subplots(1, 2, figsize=(15, 40))
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(img_seg)
    plt.show()
    return img_seg
show_segmentation(img, pixels)
