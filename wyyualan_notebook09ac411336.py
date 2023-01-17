import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pylab as plt

from random import shuffle



import os

import cv2
train_path = "../input/train"



ROWS = 128

COLS = 128

CHANNELS = 3



images = [img for img in os.listdir(train_path)]

images_dog = [img for img in os.listdir(train_path) if "dog" in img]

images_cat = [img for img in os.listdir(train_path) if "cat" in img]
