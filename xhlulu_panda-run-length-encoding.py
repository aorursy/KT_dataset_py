import os



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import skimage.io

from PIL import Image

import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
def rle_encode(img):

    '''

    Source: https://www.kaggle.com/paulorzp/run-length-encode-and-decode

    --

    img: numpy array, 1 - mask, 0 - background

    Returns run length as string formated

    '''

    pixels = img.flatten()

    pixels = np.concatenate([[0], pixels, [0]])

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    runs[1::2] -= runs[::2]

    runs = runs.astype(str).tolist()

    return ' '.join(runs)





def rle_decode(mask_rle, shape):

    '''

    Source: https://www.kaggle.com/paulorzp/run-length-encode-and-decode

    --

    mask_rle: run-length as string formated (start length)

    shape: (height,width) of array to return 

    Returns numpy array, 1 - mask, 0 - background



    '''

    s = mask_rle.split()

    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    starts -= 1

    ends = starts + lengths

    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):

        img[lo:hi] = 1

    return img.reshape(shape)
data_dir = '/kaggle/input/prostate-cancer-grade-assessment/train_images/'

mask_dir = '/kaggle/input/prostate-cancer-grade-assessment/train_label_masks/'

mask_files = os.listdir(mask_dir)
mask = skimage.io.MultiImage(mask_dir + mask_files[0])

img = skimage.io.MultiImage(data_dir + mask_files[0].replace("_mask", ""))



fig, axes = plt.subplots(1, 5, figsize=(15, 3))



axes[0].imshow(mask[-1][:, :, 0] == 1)

axes[1].imshow(mask[-1][:, :, 0] == 2)

axes[2].imshow(mask[-1][:, :, 0] == 3)

axes[3].imshow(mask[-1][:, :, 0] == 4)

axes[4].imshow(mask[-1][:, :, 0] == 5)
encs = []

shapes = []



file = mask_files[0]



for file in tqdm(mask_files):



    mask = skimage.io.MultiImage(mask_dir + file)

    enc = []



    for n in range(1,6):

        enc.append(rle_encode(mask[-1][:, :, 0] == n))



    encs.append(enc)

    shapes.append(mask[-1].shape)
rle_df = pd.DataFrame(encs,  columns=[f'mask_{i}' for i in range(1, 6)])

rle_df['file'] = mask_files

rle_df['shapes'] = shapes

rle_df.head()
rle_df.to_csv('train_masks_rle.csv')