# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
import matplotlib.pyplot as plt
%matplotlib inline
from PIL import Image
from PIL import ImageFilter
import multiprocessing
import random; random.seed(2016);
import cv2
import re
import os, glob
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt
from skimage import io

def compare(*images, **kwargs):
    """
    Utility function to display images side by side.
    
    Parameters
    ----------
    image0, image1, image2, ... : ndarrray
        Images to display.
    labels : list
        Labels for the different images.
    """
    f, axes = plt.subplots(1, len(images), **kwargs)
    axes = np.array(axes, ndmin=1)
    
    labels = kwargs.pop('labels', None)
    if labels is None:
        labels = [''] * len(images)
    
    for n, (image, label) in enumerate(zip(images, labels)):
        axes[n].imshow(image, interpolation='nearest', cmap='gray')
        axes[n].set_title(label)
        axes[n].axis('off')
    
    plt.tight_layout()
pano_imgs = io.ImageCollection('../input/image/image/stich1/*')
compare(*pano_imgs, figsize=(30, 20))


def stich(img1,img2):
    #img1 = cv2.imread(files.path[i1])#pano_imgs[i1]
    #img2 = cv2.imread(files.path[i2])#pano_imgs[i2]
    # Convert images to grayscale
    im1_gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    # Find size of image1
    sz = img1.shape
    # Define the motion model
    warp_mode = cv2.MOTION_TRANSLATION
    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    # Specify the number of iterations.
    number_of_iterations = 5000;
    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10;
    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
    # Run the ECC algorithm. The results are stored in warp_matrix.
    cc, warp_matrix = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria)
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        # Use warpPerspective for Homography 
        im2_aligned = cv2.warpPerspective (img2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else :
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(img2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
        #im3 = cv2.warpPerspective(img1[:, :, ::-1], wrap_matrix, img1.shape[:2][::-1])
    print(im2_aligned.shape)
    print(img1.shape)
    print(img2.shape)
    ## (1) Convert to gray, and threshold
    gray = cv2.cvtColor(im2_aligned, cv2.COLOR_BGR2GRAY)
    off=0
    for i in range(0,22):
        if(gray[115][i] == 0):
            off=off+1
    print(off)
    #compare(im2_aligned , img1, img2 , np.concatenate((img1[:,:-2,:], img2[:,off-1:-1,:]), axis=1), figsize=(15, 10))
    #vis = np.concatenate((img1[:,:-2,:], img2[:,off:-1,:]), axis=1)
    return off
print(len(pano_imgs))
offs=[0 for i in range(131)]
j=0
for i in range(15,130):
    offs[i] = stich(pano_imgs[i],pano_imgs[i+1])
    j=j+1
print(offs)
img = pano_imgs[15]
for l in range(15,130):
    im = pano_imgs[l]
    off = offs[l]
    vis = np.concatenate((img[:,:off-34,:], im[:,:,:]), axis=1)
    img = vis
compare(img, figsize=(15, 10))
cv2.imwrite('stiched.jpeg',img)

pano_imgs1 = io.ImageCollection('../input/image/image/stitch/*')
compare(*pano_imgs1, figsize=(30, 20))
print(len(pano_imgs1))
img = pano_imgs[0]
for l in range(1,59):
    im = pano_imgs1[l]
    vis = np.concatenate((img, im), axis=1)
    img = vis
compare(img, figsize=(15, 10))