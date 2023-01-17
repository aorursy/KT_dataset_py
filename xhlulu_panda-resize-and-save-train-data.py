import os



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import openslide

import matplotlib.pyplot as plt

from PIL import Image

import cv2

from tqdm.notebook import tqdm

import skimage.io

from skimage.transform import resize, rescale
train_labels = pd.read_csv('/kaggle/input/prostate-cancer-grade-assessment/train.csv')
train_labels.head()
data_dir = '/kaggle/input/prostate-cancer-grade-assessment/train_images/'
mask_dir = '/kaggle/input/prostate-cancer-grade-assessment/train_label_masks/'

mask_files = os.listdir(mask_dir)
img_id = train_labels.image_id[0]

path = data_dir + img_id + '.tiff'
%time biopsy = openslide.OpenSlide(path)

%time biopsy2 = skimage.io.MultiImage(path)
%timeit img = biopsy.get_thumbnail(size=(512, 512))

%timeit out = resize(biopsy2[-1], (512, 512))

%timeit out = cv2.resize(biopsy2[-1], (512, 512))

%timeit out = Image.fromarray(biopsy2[-1]).resize((512, 512))
out = cv2.resize(biopsy2[-1], (512, 512))



%timeit Image.fromarray(out).save(img_id+'.png')

%timeit cv2.imwrite(img_id+'.png', out)
mask = skimage.io.MultiImage(mask_dir + mask_files[1])

img = skimage.io.MultiImage(data_dir + mask_files[1].replace("_mask", ""))
mask[-1].shape, img[-1].shape
save_dir = "/kaggle/train_images/"

os.makedirs(save_dir, exist_ok=True)
for img_id in tqdm(train_labels.image_id):

    load_path = data_dir + img_id + '.tiff'

    save_path = save_dir + img_id + '.png'

    

    biopsy = skimage.io.MultiImage(load_path)

    img = cv2.resize(biopsy[-1], (512, 512))

    cv2.imwrite(save_path, img)
save_mask_dir = '/kaggle/train_label_masks/'

os.makedirs(save_mask_dir, exist_ok=True)
for mask_file in tqdm(mask_files):

    load_path = mask_dir + mask_file

    save_path = save_mask_dir + mask_file.replace('.tiff', '.png')

    

    mask = skimage.io.MultiImage(load_path)

    img = cv2.resize(mask[-1], (512, 512))

    cv2.imwrite(save_path, img)
!tar -czf train_images.tar.gz ../train_images/*.png

!tar -czf train_label_masks.tar.gz ../train_label_masks/*.png