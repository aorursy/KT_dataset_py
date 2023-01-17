import os

import glob

import math

import numpy as np

import pandas as pd

import cv2

from matplotlib import pyplot as plt

%matplotlib inline
train = pd.read_csv("../input/train.csv")

train.sort_values(by=['species', 'id'], inplace=True)

train.species = train.species.astype('category')

train.reset_index(drop=True, inplace=True)

train.head()
img_num = 1584

imgs = []

img_heights = []

img_widths = []



for i in range(1, img_num+1):

    path = os.path.join('..', 'input', 'images', str(i) + '.jpg')

    img = cv2.imread(path, 0)

    img_height, img_width = img.shape[:2]

    imgs.append(img)

    img_heights.append(img_height)

    img_widths.append(img_width)
for sp in train.species.unique():

    train_by_species = train[train.species==sp]

    train_by_species.reset_index(drop=True, inplace=True)

    for i in range(3):

        plt.subplot(1, 3, i + 1)

        plt.imshow(imgs[train_by_species.id[i]-1], cmap='gray')

        plt.title(train_by_species.species[i])

        plt.xticks(())

    plt.tight_layout()

    plt.show()