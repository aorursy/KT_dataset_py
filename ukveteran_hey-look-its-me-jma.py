# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
from matplotlib import pyplot
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# load image
from matplotlib import pyplot

from PIL import Image
image = Image.open('../input/johar-pic/79669868_10158168171086929_2402259235392978944_n.jpg')
# horizontal flip
hoz_flip = image.transpose(Image.FLIP_LEFT_RIGHT)
# vertical flip
ver_flip = image.transpose(Image.FLIP_TOP_BOTTOM)
# plot all three images using matplotlib
pyplot.subplot(311)
pyplot.imshow(image)
pyplot.subplot(312)
pyplot.imshow(hoz_flip)
pyplot.subplot(313)
pyplot.imshow(ver_flip)
pyplot.show()
# load image
image = Image.open('../input/johar-pic/79669868_10158168171086929_2402259235392978944_n.jpg')
# plot original image
pyplot.subplot(311)
pyplot.imshow(image)
# rotate 45 degrees
pyplot.subplot(312)
pyplot.imshow(image.rotate(45))
# rotate 90 degrees
pyplot.subplot(313)
pyplot.imshow(image.rotate(90))
pyplot.show()
#rotates 270 degrees
pyplot.imshow(image.rotate(270))
pyplot.show()
#Codes from Bulent Siyah https://www.kaggle.com/bulentsiyah/learn-opencv-by-examples-with-python
#Sharpening

image = cv2.imread('../input/johar-pic/79669868_10158168171086929_2402259235392978944_n.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(20, 20))
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(image)

# Create our shapening kernel, we don't normalize since the 
# the values in the matrix sum to 1
kernel_sharpening = np.array([[-1,-1,-1], 
                              [-1,9,-1], 
                              [-1,-1,-1]])

# applying different kernels to the input image
sharpened = cv2.filter2D(image, -1, kernel_sharpening)


plt.subplot(1, 2, 2)
plt.title("Image Sharpening")
plt.imshow(sharpened)

plt.show()
from PIL import Image
# load image
image = Image.open('../input/johar-pic/79669868_10158168171086929_2402259235392978944_n.jpg')
# create a cropped image
cropped = image.crop((100, 100, 200, 200))
# show cropped image
pyplot.imshow(cropped)
from PIL import Image

image = Image.open("../input/johar-pic/79669868_10158168171086929_2402259235392978944_n.jpg")
image