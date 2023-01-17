import numpy as np

import pandas as pd

from skimage.color import rgb2gray

import matplotlib.pyplot as plt

%matplotlib inline
img_list = ['d527d08861e41c7b.jpg', '82f4f279c9a96ee8.jpg', '309b0f22a3f7efe0.jpg', 'a57366e38050b227.jpg']

fig = plt.figure(figsize=(16, 16))

for i in range(4):

    x = fig.add_subplot(2, 2, i+1)

    image = plt.imread('/kaggle/input/test/'+img_list[i])

    x.set_title("{image} ({shape[0]},{shape[1]})".format(image=img_list[i],shape=image.shape))

    plt.imshow(image)
fig = plt.figure(figsize=(16, 16))

for i in range(4):

    x = fig.add_subplot(2, 2, i+1)

    image = plt.imread('/kaggle/input/test/'+img_list[i])

    gray = rgb2gray(image)

    x.set_title("{image} ({shape[0]},{shape[1]})".format(image=img_list[i],shape=gray.shape))

    plt.imshow(gray, cmap='gray')
def to_2regions(image):

    gray = rgb2gray(image)

    m = gray.mean()

    for i in range(image.shape[0]):

        for j in range(image.shape[1]):    

            gray[i,j] = int(gray[i,j] > m)

    return gray
fig = plt.figure(figsize=(16, 16))

for i in range(4):

    x = fig.add_subplot(2, 2, i+1)

    image = plt.imread('/kaggle/input/test/'+img_list[i])

    r2_gray = to_2regions(image)

    x.set_title("{image} ({shape[0]},{shape[1]})".format(image=img_list[i],shape=r2_gray.shape))

    plt.imshow(r2_gray, cmap='gray')
def to_4regions(image):

    g_image = rgb2gray(image)

    m = g_image.mean()

    m1 = 0.5 * m

    m2 = 0.25 * m

    for i in range(image.shape[0]):

        for j in range(image.shape[1]):

            pxl = g_image[i,j]

            if pxl > m:

                g_image[i,j] = 3

            elif pxl > m1:

                g_image[i,j] = 2

            elif pxl > m2:

                g_image[i,j] = 1

            else:

                g_image[i,j] = 0            

    return g_image
fig = plt.figure(figsize=(16, 16))

for i in range(4):

    x = fig.add_subplot(2, 2, i+1)

    image = plt.imread('/kaggle/input/test/'+img_list[i])

    r2_gray = to_4regions(image)

    x.set_title("{image} ({shape[0]},{shape[1]})".format(image=img_list[i],shape=r2_gray.shape))

    plt.imshow(r2_gray, cmap='gray')