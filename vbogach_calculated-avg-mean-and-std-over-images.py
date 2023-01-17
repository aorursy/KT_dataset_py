# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

from tqdm import tqdm

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
def calc_avg_mean_std(img_names, img_root):

    mean_sum = np.array([0., 0., 0.])

    std_sum = np.array([0., 0., 0.])

    n_images = len(img_names)

    for img_name in tqdm(img_names):

        img = cv2.imread(img_root + img_name)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mean, std = cv2.meanStdDev(img)

        mean_sum += np.squeeze(mean)

        std_sum += np.squeeze(std)

    return (mean_sum / n_images, std_sum / n_images)
train_img_root = '../input/train_images/'

train_img_names = os.listdir(train_img_root)

train_mean, train_std = calc_avg_mean_std(train_img_names, train_img_root)

train_mean, train_std
test_img_root = '../input/test_images/'

test_img_names = os.listdir(test_img_root)

test_mean, test_std = calc_avg_mean_std(test_img_names, test_img_root)

test_mean, test_std
train_mean / 255., train_std / 255.
test_mean / 255., test_std / 255.