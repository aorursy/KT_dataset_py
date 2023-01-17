# load package

import cv2

import numpy as np

from math import pi

import matplotlib.pyplot as plt

%matplotlib inline 
import colorsys

# convert the bgr to hsv space

def BGR_TO_HSV(img):

    with np.errstate(divide='ignore', invalid='ignore'):

        bgr = np.int32(cv2.split(img))

        blue = bgr[0]

        green = bgr[1]

        red = bgr[2]

        h = np.zeros(blue.shape)

        s = np.zeros(blue.shape)

        v = np.zeros(blue.shape)

        for i in range(blue.shape[0]):

            for j in range(blue.shape[1]):

                th,ts,tv = colorsys.rgb_to_hsv(red[i,j], green[i,j], blue[i,j])

                h[i,j] = (th)

                s[i,j] = (ts)

                v[i,j] = (tv)

        return h,s,v
img = cv2.imread('../input/PandaTest.jpeg')

im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(14,14))

plt.imshow(im_rgb)

plt.show()
h,s,v = BGR_TO_HSV(img)
v[0,0]