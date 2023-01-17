import sys

sys.path.insert(0, '../input/siim-acr-pneumothorax-segmentation')

import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pathlib import Path
DATA_PATH = Path("../input/siim-acr-pneumothorax-segmentation-data/pneumothorax")

print(DATA_PATH)
TRAIN = DATA_PATH / "dicom-images-train"

TEST = DATA_PATH / "dicom-images-test"
import glob2
print(TRAIN)
train = glob2.glob(os.path.join(TRAIN, '**/*.dcm'))
count= 0

for f in train:

    #print(f.split('/')[7:])

    count = count + 1

print('Total training file: ', count)
test = glob2.glob(os.path.join(TEST, '**/*.dcm'))
count = 0

for f in test:

    #print(f.split('/')[7:])

    count = count + 1

print('Total test file: ', count)
import pydicom
images_dict = {}

count = 0

for f in train:

    #print(f.split('/')[7:])

    imageid = f.split('/')[7:][0]

    imageid = imageid.strip('.dcm')

    #print(imageid)

    images_dict[imageid] = f 

    count = count+1

print('Training images: ', count)
print(images_dict['1.2.276.0.7230010.3.1.4.8323329.10026.1517875221.22915'])
import matplotlib.pyplot as plt
fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(30,30))

i = 0

for key, val in images_dict.items():

    img = pydicom.dcmread(images_dict[key])

    ax[i].imshow(img.pixel_array, cmap=plt.cm.bone)

    i = i + 1

    if i == 4: break
import random

fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(30,30))

i = 0

for key in random.sample(list(images_dict.keys()),4):

    img = pydicom.dcmread(images_dict[key])

    ax[i].imshow(img.pixel_array, cmap=plt.cm.bone)

    i = i + 1
no_mask_images=['1.2.276.0.7230010.3.1.4.8323329.11566.1517875233.640521',

'1.2.276.0.7230010.3.1.4.8323329.11104.1517875231.169401',

'1.2.276.0.7230010.3.1.4.8323329.31801.1517875156.929061',

'1.2.276.0.7230010.3.1.4.8323329.11584.1517875233.731531',

'1.2.276.0.7230010.3.1.4.8323329.11557.1517875233.601090',

'1.2.276.0.7230010.3.1.4.8323329.3352.1517875177.433385',

'1.2.276.0.7230010.3.1.4.8323329.14557.1517875252.690062',

'1.2.276.0.7230010.3.1.4.8323329.4373.1517875182.554858',

'1.2.276.0.7230010.3.1.4.8323329.2563.1517875173.431928',

'1.2.276.0.7230010.3.1.4.8323329.12062.1517875237.179186',

'1.2.276.0.7230010.3.1.4.8323329.4468.1517875183.20323',

'1.2.276.0.7230010.3.1.4.8323329.4843.1517875185.73985',

'1.2.276.0.7230010.3.1.4.8323329.10231.1517875222.737143',

'1.2.276.0.7230010.3.1.4.8323329.10407.1517875223.567351',

'1.2.276.0.7230010.3.1.4.8323329.3089.1517875176.36192',

'1.2.276.0.7230010.3.1.4.8323329.11577.1517875233.694347',

'1.2.276.0.7230010.3.1.4.8323329.2309.1517875172.75133',

'1.2.276.0.7230010.3.1.4.8323329.4134.1517875181.277174',

'1.2.276.0.7230010.3.1.4.8323329.13415.1517875245.218707',

'1.2.276.0.7230010.3.1.4.8323329.10599.1517875224.488727',

'1.2.276.0.7230010.3.1.4.8323329.1068.1517875166.144255',

'1.2.276.0.7230010.3.1.4.8323329.13620.1517875246.884737',

'1.2.276.0.7230010.3.1.4.8323329.4996.1517875185.888529',

'1.2.276.0.7230010.3.1.4.8323329.5278.1517875187.330082',

'1.2.276.0.7230010.3.1.4.8323329.2630.1517875173.773726',

'1.2.276.0.7230010.3.1.4.8323329.3714.1517875179.128897',

'1.2.276.0.7230010.3.1.4.8323329.5543.1517875188.726955',

'1.2.276.0.7230010.3.1.4.8323329.3321.1517875177.247887',

'1.2.276.0.7230010.3.1.4.8323329.10362.1517875223.377845',

'1.2.276.0.7230010.3.1.4.8323329.2187.1517875171.557615',

'1.2.276.0.7230010.3.1.4.8323329.3791.1517875179.436805',

'1.2.276.0.7230010.3.1.4.8323329.5087.1517875186.354925',

'1.2.276.0.7230010.3.1.4.8323329.32688.1517875161.809571',

'1.2.276.0.7230010.3.1.4.8323329.11215.1517875231.757436',

'1.2.276.0.7230010.3.1.4.8323329.32302.1517875159.778024',

'1.2.276.0.7230010.3.1.4.8323329.2083.1517875171.71387',

'1.2.276.0.7230010.3.1.4.8323329.13378.1517875244.961609']
for id in no_mask_images:

    del images_dict[id]
import cv2

import zipfile

print(cv2.__version__)
out_sz = (256,256)
OUTPUT = '../working/train-images/'

if not os.path.exists(OUTPUT):

    os.mkdir(OUTPUT)
i = 0

with zipfile.ZipFile(os.path.join(OUTPUT, 'train-images-256.zip'), 'w') as zip:

    for key in images_dict.keys():

        img = pydicom.dcmread(images_dict[key])

        file = key+'.png'

        #print(file)

        _, png = cv2.imencode('.png', cv2.resize(img.pixel_array, out_sz))

        #print(png)

        zip.writestr(file, png)

        i = i + 1

print('Total images written: ', i)   