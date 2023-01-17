import numpy as np
import pandas as pd
import os
from os.path import join
import glob
import cv2
import random 

import matplotlib.pyplot as plt
train_path = '../input/stage1_train/'
test_path = '../input/stage1_test/'

seed = 42
random.seed = seed
np.random.seed = seed

import pathlib
train_path = pathlib.Path(train_path).glob('*/images/*.png')
print(train_path)

train_sorted = sorted([i for i in train_path])
print(len(train_sorted))
# show a random image by converting it to grayscale
im = np.random.choice(train_sorted)
im = cv2.imread(str(im), cv2.IMREAD_GRAYSCALE)
plt.imshow(im)
# using opencv otsu
# ret, thresh_val = cv2.threshold(im,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret, thresh_val = cv2.threshold(im, 100, 255, cv2.THRESH_OTSU)
print(ret)
print(thresh_val)

# # show the original image and the image after threshoulding
# fig, ax = plt.subplots(1, 2, figsize=(10,10))
# ax[0].imshow(im)
# ax[1].imshow(thresh_val)

from skimage.filters import threshold_otsu
# alternative threshoulding 
thresh_val_1 = threshold_otsu(im)
print('Otsu Threshold',thresh_val_1)

# Compute masks using threshould
mask = np.where(im > thresh_val_1, 1, 0) # masks are encoded as 1 and others as 0
print('Original image shape',im.shape)
print('Mast shape', mask.shape)

# show original and masked images
fig, ax = plt.subplots(1, 2, figsize=(10,10))
ax[0].imshow(im)
ax[1].imshow(mask)
display_mask = np.where(mask, mask, np.nan)
print('Display mask shape:', display_mask.shape)

if np.sum(mask==0) < np.sum(mask==1):
    mask = np.where(mask, 0, 1)

# Plot images 
plt.figure(figsize=(10,4))
plt.subplot(1,2,2)
plt.imshow(im, cmap='gray')
plt.imshow(mask, cmap='rainbow', alpha=0.5)
plt.axis('off')
plt.title('Image with Mask')

# image without mask
plt.figure(figsize=(10,4))
plt.subplot(1,2,2)
plt.imshow(im, cmap='gray')
plt.imshow(display_mask, cmap='rainbow', alpha=0.5)
plt.axis('off')
plt.title('Image without Mask')

from scipy import ndimage
labels, nlabels = ndimage.label(mask)

print('No of labels found = ', nlabels)

label_arrays = []
for label_num in range(1, nlabels+1):
    label_mask = np.where(labels == label_num, 1, 0)
    label_arrays.append(label_mask)

print('{} separate objects detected.'.format(nlabels))
# flatten masks 
label_mask.T.flatten()

def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return " ".join([str(i) for i in run_lengths])

print('RLE Encoding for the current mask : {}'.format(rle_encoding(label_mask)))
# example of a flattened mask image
x = np.where(label_mask.T.flatten()==1)[0]
x = np.array(x)
x.shape
