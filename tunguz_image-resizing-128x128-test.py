# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os

import gc

import json

import math

import cv2

import PIL

from PIL import Image

import numpy as np



import matplotlib.pyplot as plt

import pandas as pd



import scipy

from tqdm import tqdm

%matplotlib inline

from keras.preprocessing import image
imSize = 128
test_df = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')

print(test_df.shape)

test_df.head()
def preprocess_image(image_path, desired_size=imSize):

    im = Image.open(image_path)

    im = im.resize((desired_size, )*2, resample=Image.LANCZOS)

    

    return im
N = test_df.shape[0]

x_test = np.empty((N, imSize, imSize, 3), dtype=np.uint8)

for i, image_id in enumerate(tqdm(test_df['image_name'])):

    x_test[i, :, :, :] = preprocess_image(

        f'../input/siim-isic-melanoma-classification/jpeg/test/{image_id}.jpg'

    )
np.save('x_test_128', x_test)