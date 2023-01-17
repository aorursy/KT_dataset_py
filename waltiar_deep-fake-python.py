# Loading Chunk



import scipy as sci

import os

import numpy as np

import pandas as pd

import matplotlib.pylab as plt

import cv2

plt.style.use('ggplot')

from IPython.display import Video

from IPython.display import HTML



sample_submission = pd.read_csv("../input/deepfake-detection-challenge/sample_submission.csv")
print(sample_submission)
train_sample_metadata = pd.read_json('../input/deepfake-detection-challenge/train_sample_videos/metadata.json').T

train_sample_metadata.head()
train_sample_metadata.groupby('label')['label'].count().plot(figsize=(15, 5), kind='bar', title='Distribution of Labels in the Training Set')

plt.show()