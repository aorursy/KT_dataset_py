import gc

import os

import glob

import zipfile

import warnings

import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from tqdm import tqdm



import cv2

import PIL

from PIL import ImageOps, ImageFilter, ImageDraw
DATA_PATH = '../input/'

os.listdir(DATA_PATH)
# 이미지 폴더 경로

TRAIN_IMG_PATH = os.path.join(DATA_PATH, 'train')

TEST_IMG_PATH = os.path.join(DATA_PATH, 'test')



# CSV 파일 경로

df_train = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))

df_test = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))

df_class = pd.read_csv(os.path.join(DATA_PATH, 'class.csv'))
df_train.head()
df_test.head()
def crop_boxing_img(img_name, margin=16) :

    if img_name.split('_')[0] == "train" :

        PATH = TRAIN_IMG_PATH

        data = df_train

    elif img_name.split('_')[0] == "test" :

        PATH = TEST_IMG_PATH

        data = df_test

        

    img = PIL.Image.open(os.path.join(PATH, img_name))

    pos = data.loc[data["img_file"] == img_name, \

                   ['bbox_x1','bbox_y1', 'bbox_x2', 'bbox_y2']].values.reshape(-1)



    width, height = img.size

    x1 = max(0, pos[0] - margin)

    y1 = max(0, pos[1] - margin)

    x2 = min(pos[2] + margin, width)

    y2 = min(pos[3] + margin, height)

    

    if abs(pos[2] - pos[0]) > width or abs(pos[3] - pos[1]) > height:

        print(f'{img_name} is wrong bounding box, img size: {img.size},  bbox_x1: {pos[0]}, bbox_x2: {pos[2]}, bbox_y1: {pos[1]}, bbox_y2: {pos[3]}')

        return img



    return img.crop((x1,y1,x2,y2))
for i, row in df_train.iterrows():

    cropped = crop_boxing_img(row['img_file'])

    cropped.save(row['img_file'])
for i, row in df_test.iterrows():

    cropped = crop_boxing_img(row['img_file'])

    cropped.save(row['img_file'])
tmp_imgs = df_train['img_file'][100:105]

plt.figure(figsize=(12,20))



for num, f_name in enumerate(tmp_imgs):

    img = PIL.Image.open(os.path.join(TRAIN_IMG_PATH, f_name))

    plt.subplot(5, 2, 2*num + 1)

    plt.title(f_name)

    plt.imshow(img)

    plt.axis('off')

    

    img_crop = PIL.Image.open(f_name)

    plt.subplot(5, 2, 2*num + 2)

    plt.title(f_name + ' cropped')

    plt.imshow(img_crop)

    plt.axis('off')
tmp_imgs = df_test['img_file'][100:105]

plt.figure(figsize=(12,20))



for num, f_name in enumerate(tmp_imgs):

    img = PIL.Image.open(os.path.join(TEST_IMG_PATH, f_name))

    plt.subplot(5, 2, 2*num + 1)

    plt.title(f_name)

    plt.imshow(img)

    plt.axis('off')

    

    img_crop = PIL.Image.open(f_name)

    plt.subplot(5, 2, 2*num + 2)

    plt.title(f_name + ' cropped')

    plt.imshow(img_crop)

    plt.axis('off')
with zipfile.ZipFile('train_crop.zip','w') as zip: 

        # writing each file one by one 

        for file in glob.glob('train*.jpg'): 

            zip.write(file)
with zipfile.ZipFile('test_crop.zip','w') as zip: 

        # writing each file one by one 

        for file in glob.glob('test*.jpg'): 

            zip.write(file)
!rm -rf *.jpg