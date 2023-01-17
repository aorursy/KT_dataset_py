# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# Any results you write to the current directory are saved as output.
import cv2

# normalize an image for training purposes
def normalize(x):
    return (x.astype(float) - 128)/128

# open, resize and normalize an image
def preprocess(img_name, size):
    img = cv2.imread(img_name)
    img = cv2.resize(img, size)
    img = normalize(img)
    
    return img
    
images_folder = '../input/stage_1_test_images/'
image_names = os.listdir(images_folder)

image_name = images_folder+image_names[0]

print (image_name)
                         
from matplotlib import pyplot

# read and show the original image
original_image = cv2.imread(image_name)

pyplot.imshow(original_image)
# preprocess an image and show it
processed_image = preprocess(image_name, (150,150))

pyplot.imshow(processed_image)
