# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

from PIL import Image

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



def display_images(X, rows=5, columns=5, cmap="gray", h=128,w=128):

    """ Utility function to display images

    """

    fig, ax = plt.subplots(rows,columns, figsize=(8,8))

    for row in range(rows):

        for column in range(columns):

            ax[row][column].imshow(X[(row*columns)+column].reshape(h,w), cmap=cmap)

            ax[row][column].set_axis_off()
df = pd.read_csv("../input/sample/sample.csv")

df.head()
X = df.iloc[:,1:].values.reshape(len(df),128,128,1)

y = df.iloc[:,0].values



X.shape
display_images(X)
from keras.preprocessing.image import ImageDataGenerator

idg = ImageDataGenerator(

    rotation_range=30,

    zoom_range = 0.3, 

    width_shift_range=0.1,

    height_shift_range=0.1,

    horizontal_flip=True,

    vertical_flip=True,

    brightness_range=[0.9,1.01],

    fill_mode='constant', cval=255

)
image_data = idg.flow(X, y,batch_size=25).next()

display_images(image_data[0])
image_data[0][0].shape, image_data[1][0]