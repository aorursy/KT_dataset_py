import os

import imageio

import tensorflow as tf

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# load the train_csv file with the image properties

base_dir = '../input/landmark-recognition-2020/'

train_csv = pd.read_csv('../input/landmark-recognition-multiprocessing-image-size/train_featured.csv')
# This swaps the values of xsize with ysize columns as they were assigned wrongly in a previous kernel.

ys_temp = train_csv.xsize.copy()

train_csv.xsize = train_csv.ysize

train_csv.ysize = ys_temp

del ys_temp
train_csv.head()
g = sns.jointplot(x="xsize", y="ysize", data=train_csv)
print('Range of xsize and ysize:')

print('---'*8)

print(train_csv.xsize.min(), '<= xsize <=', train_csv.xsize.max())

print(train_csv.ysize.min(), '<= ysize <=', train_csv.ysize.max())
print('The most commonly observed 3 values of xsize and ysize:')

xcount = train_csv.xsize.value_counts()

ycount = train_csv.ysize.value_counts()

print(xcount[:3])

print(ycount[:3])
train_csv[(train_csv.ysize < 100) & (train_csv.xsize > 600)]
def load_image(idx):

    impath = base_dir + 'train/' + '/'.join(list(idx[:3])) + '/' + idx + '.jpg'

    return imageio.imread(impath)
plt.imshow(load_image(train_csv.loc[43027, 'id']))

plt.show()
plt.imshow(load_image(train_csv.loc[491551, 'id']))

plt.show()
plt.imshow(load_image(train_csv.loc[1289336, 'id']))

plt.show()
train_csv[(train_csv.xsize < 200) & (train_csv.ysize > 600)]
plt.imshow(load_image(train_csv.loc[338829, 'id']))

plt.show()
plt.imshow(load_image(train_csv.loc[546929, 'id']))

plt.show()
plt.imshow(load_image(train_csv.loc[965744, 'id']))

plt.show()