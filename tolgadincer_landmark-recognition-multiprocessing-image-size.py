import os

import time

import tqdm

import imageio

import numpy as np

import pandas as pd

import tensorflow as tf

from multiprocessing import Pool

import seaborn as sns; sns.set(style="white", color_codes=True)

import matplotlib.pyplot as plt

%matplotlib inline
base_dir = '../input/landmark-recognition-2020/'

train_csv = pd.read_csv(base_dir + 'train.csv')

sample_submission = pd.read_csv(base_dir + 'sample_submission.csv')
train_csv.head()
train_csv.info()
def get_image_features(lid):

    impath = base_dir + 'train/' + '/'.join(list(lid[:3])) + '/' + lid + '.jpg'

    im = imageio.imread(impath)

    xsize = im.shape[-3]

    ysize = im.shape[-2]

    depth = im.shape[-1]

    return xsize, ysize, depth
with Pool(4) as p:

    r = list(tqdm.tqdm(p.imap(get_image_features, train_csv.id), total=1000))
train_csv['xsize'] = np.array(r).T[0]

train_csv['ysize'] = np.array(r).T[1]

train_csv['depth'] = np.array(r).T[2]
train_csv.to_csv('train_featured.csv', index=False)
g = sns.jointplot(x="xsize", y="ysize", data=train_csv)