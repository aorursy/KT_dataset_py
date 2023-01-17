import os, cv2, re, random
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
from keras import layers, models, optimizers
from keras import backend as K
from sklearn.model_selection import train_test_split

test_data = pd.read_csv('../input/test.csv')
train_data = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
test.head()
test.shape
train = pd.read_csv('../input/train.csv')
train.shape
