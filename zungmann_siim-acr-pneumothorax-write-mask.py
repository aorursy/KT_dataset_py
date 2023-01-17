import sys

sys.path.insert(0, '../input/siim-acr-pneumothorax-segmentation')

import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pathlib import Path

from mask_functions import *
DATA_PATH = Path("../input/siim-acr-pneumothorax-segmentation-data/pneumothorax")

print(DATA_PATH)
print(os.listdir(DATA_PATH))
TRAIN = DATA_PATH / "dicom-images-train"

TEST = DATA_PATH / "dicom-images-test"

MASK = DATA_PATH / "train-rle.csv"
mask_rle_df = pd.read_csv(MASK)

mask_rle_df.describe()
mask_rle_df.info()
mask_rle_df = mask_rle_df.rename(columns={' EncodedPixels': 'EncodedPixels'})

mask_rle_df.info()
img_mask_df = mask_rle_df[mask_rle_df['EncodedPixels'] != ' -1'].groupby('ImageId').count()

img_mask_df.head()
img_mask_df = img_mask_df.rename(columns={'EncodedPixels': 'NMasks'})

img_mask_df['EncodedMasks']=''

img_mask_df.head()
i = 0

for index, row in img_mask_df.iterrows():

    #print(imageid)

    rles = mask_rle_df.loc[mask_rle_df['ImageId']==index, 'EncodedPixels'].values

    #print(rles)

    img_mask_df.at[index,'EncodedMasks'] = rles

    i = i+1

print('Total image: ', i)
img_nomask_df = mask_rle_df[mask_rle_df['EncodedPixels'] == ' -1'].groupby('ImageId').sum()

img_nomask_df.head()
import cv2

import zipfile
OUTPUT = '../working/masks/'

if not os.path.exists(OUTPUT):

    os.mkdir(OUTPUT)
out_size = (256,256)

i = 0

j = 0

with zipfile.ZipFile(os.path.join(OUTPUT, 'mask.zip'), 'w') as zip:

    for index, row in img_mask_df.iterrows():

        rles = row['EncodedMasks']

        mask = np.zeros((1024,1024))

        for rle in rles:

            mask = mask + rle2mask(rle, 1024,1024).T

        file = index+'.png'

        _, png = cv2.imencode('.png', cv2.resize(mask, out_size))

        zip.writestr(file, png)

        i = i + 1

    for index, row in img_nomask_df.iterrows():

        mask = np.zeros((1024,1024))

        file = index+'.png'

        _, png = cv2.imencode('.png', cv2.resize(mask, out_size))

        zip.writestr(file, png)

        j = j + 1

print('Total mask files written: ', i+j)
print(os.listdir('../working/masks'))