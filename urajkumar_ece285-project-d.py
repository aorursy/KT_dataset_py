import os

import json

import gc

import cv2

import keras

from keras.models import load_model

import tensorflow as tf

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import sys
import pandas as pd

df = pd.read_csv('../input/severstal-steel-defect-detection/sample_submission.csv')

df.set_index('ImageId_ClassId', inplace=True)



df_submit = pd.read_csv("../input/ece285-sub/unet_res.csv")

df_submit.set_index('ImageId_ClassId', inplace=True)

9

for name, row in df_submit.iterrows():

    df.loc[name] = row



df.reset_index(inplace=True)

df.to_csv('submission.csv', index=False)