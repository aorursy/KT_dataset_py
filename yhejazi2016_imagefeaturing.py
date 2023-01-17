# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt



from skimage.io import imread, imshow

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
image = imread('/kaggle/input/puppy.jpeg', as_gray=True) #, as_gray=True

image.shape

image.shape, imshow(image)

#checking image shape 

image.shape, image
#Method #1: Grayscale Pixel Values as Features

#pixel features



features = np.reshape(image, (280*211))



features.shape, features
# Method #2: Mean Pixel Value of Channels



image = imread('/kaggle/input/puppy.jpeg')

image.shape

feature_matrix = np.zeros((280,211)) 

feature_matrix.shape

for i in range(0,image.shape[0]):

    for j in range(0,image.shape[1]):

        feature_matrix[i][j] = ((int(image[i,j,0]) + int(image[i,j,1]) + int(image[i,j,2]))/3)



features = np.reshape(feature_matrix, (280*211)) 

features.shape
#Method3: Extracting Edge Features

from skimage.io import imread, imshow

from skimage.filters import prewitt_h,prewitt_v 



#reading the image 

image = imread('/kaggle/input/puppy.jpeg',as_gray=True)



#calculating horizontal edges using prewitt kernel

edges_prewitt_horizontal = prewitt_h(image)

#calculating vertical edges using prewitt kernel

edges_prewitt_vertical = prewitt_v(image)



imshow(edges_prewitt_vertical, cmap='gray')



edges_prewitt_horizontal
edges_prewitt_vertical