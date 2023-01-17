!pip install mrcnn
from shutil import copyfile

copyfile(src = "../input/nucleus-py/nucleus.py", dst = "../working/nucleus.py")
import os

import sys

import itertools

import math

import logging

import json

import re

import random

import time

import concurrent.futures

import numpy as np

import pandas as pd

import matplotlib

import matplotlib.pyplot as plt

import matplotlib.patches as patches

import matplotlib.lines as lines

from matplotlib.patches import Polygon

import imgaug

from imgaug import augmenters as iaa



#Import Mask-RCNN

from mrcnn import utils

from mrcnn import visualize

from mrcnn.visualize import display_images

from mrcnn import model as modellib

from mrcnn.model import log

from mrcnn.config import Config



import nucleus

#from "../input/nucleuspyfile" import nucleus



def rle_decode(mask_rle, shape=(768, 768)):

   

    #mask_rle: run-length as string formated (start length)

    #shape: (height,width) of array to return

    #Returns numpy array, 1 - mask, 0 - background

    

    s = mask_rle.split()

    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    starts -= 1

    ends = starts + lengths

    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)

    

    for lo, hi in zip(starts, ends):

        img[lo:hi] = 1

        

    return img.reshape(shape).T  # Needed to align to RLE direction
df_masks = pd.read_csv('../input/2019-fire171-asn11-image-segmentation-challenge-v2/train_masks.csv')

images = np.array(os.listdir('../input/2019-fire171-asn11-image-segmentation-challenge-v2/train_images/train_images'))

df_masks.head()



mask_ids = np.array(df_masks['ImageId'])

mask_pixels = np.array(df_masks['EncodedPixels'])



data_dir = "../input/2019-fire171-asn11-image-segmentation-challenge-v2/train_images/train_images"



for p in range(images.shape[0]):

    img_dir = data_dir + images[p]

    

    os.mkdir(img_dir + "/masks/")

    img_dir = data_dir + images[p] + "/images/" + images[p] + ".png"

    img = plt.imread(img_dir)

    

    
images = np.array(os.listdir('../input/2019-fire171-asn11-image-segmentation-challenge-v2/train_images/train_images/44afae184c89e6ba55985b4d341acc1ae1e8b6ef96312064e0e6e630e022b078/images'))

images[0]



img = plt.imread('../input/2019-fire171-asn11-image-segmentation-challenge-v2/train_images/train_images/44afae184c89e6ba55985b4d341acc1ae1e8b6ef96312064e0e6e630e022b078/images/44afae184c89e6ba55985b4d341acc1ae1e8b6ef96312064e0e6e630e022b078.png')

plt.imshow(img)

img.shape
# Dataset directory

DATASET_DIR = "../input/2019-fire171-asn11-image-segmentation-challenge-v2/"



# Use configuation from nucleus.py, but override

# image resizing so we see the real sizes here

class NoResizeConfig(nucleus.NucleusConfig):

    IMAGE_RESIZE_MODE = "none"

    

config = NoResizeConfig()
def get_ax(rows=1, cols=1, size=16):

    """Return a Matplotlib Axes array to be used in

    all visualizations in the notebook. Provide a

    central point to control graph sizes.

    

    Adjust the size attribute to control how big to render images

    """

    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))

    return ax
#os.listdir("../input/umd-fire171-asn6-image-segmentation-challenge-2019/")

os.listdir("../input")
# Training dataset

dataset_train = nucleus.NucleusDataset()

dataset_train.load_nucleus(DATASET_DIR, subset="train")

dataset_train.prepare()



print("Image Count: {}".format(len(dataset_train.image_ids)))

print("Class Count: {}".format(dataset_train.num_classes))

for i, info in enumerate(dataset_train.class_info):

    print("{:3}. {:50}".format(i, info['name']))

# Validation dataset

dataset_val = ShapesDataset()

dataset_val.load_shapes(50, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])

dataset_val.prepare()