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
import cv2

from PIL import Image

import matplotlib.pyplot as plt

%matplotlib inline
!pip install albumentations > /dev/null 2>&1

!pip install pretrainedmodels > /dev/null 2>&1

!pip install kekas > /dev/null 2>&1

!pip install adabound > /dev/null 2>&1
labels = pd.read_csv('../input/train.csv')

fig = plt.figure(figsize=(25, 8))

train_imgs = os.listdir("../input/train/train")

for idx, img in enumerate(np.random.choice(train_imgs, 20)):

    ax = fig.add_subplot(4, 20//4, idx+1, xticks=[], yticks=[])

    im = Image.open("../input/train/train/" + img)

    plt.imshow(im)

    lab = labels.loc[labels['id'] == img, 'has_cactus'].values[0]

    ax.set_title(f'Label: {lab}')