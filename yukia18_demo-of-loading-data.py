import os



import cv2

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



%matplotlib inline
INPUT_DIR = '../input/ailab-ml-training-0/'

TRAIN_IMAGE_DIR = 'train_images/train_images/'

TEST_IMAGE_DIR = 'test_images/test_images/'



ID = 'fname'

TARGET = 'label'
train_df = pd.read_csv(os.path.join(INPUT_DIR, 'train.csv'))

sample_submission_df = pd.read_csv(os.path.join(INPUT_DIR, 'sample_submission.csv'))
train_df.head()
sample_index = [0, 10, 100]



fig, ax = plt.subplots(1, len(sample_index))

fig.set_size_inches(4*len(sample_index), 4)



for i, idx in enumerate(sample_index):

    fname, label = train_df.loc[idx, [ID, TARGET]]

    img = cv2.imread(os.path.join(INPUT_DIR, TRAIN_IMAGE_DIR, fname))

    ax[i].imshow(img)

    ax[i].set_title(f'{fname} - label: {label}')



plt.show()
sample_index = [0, 10, 100]



fig, ax = plt.subplots(1, len(sample_index))

fig.set_size_inches(4*len(sample_index), 4)



for i, idx in enumerate(sample_index):

    fname, label = sample_submission_df.loc[idx, [ID, TARGET]]

    img = cv2.imread(os.path.join(INPUT_DIR, TEST_IMAGE_DIR, fname))

    ax[i].imshow(img)

    ax[i].set_title(f'{fname} - sample: {label}')



plt.show()
sample_submission_df.head()
sample_submission_df.to_csv('submission.csv', index=False)