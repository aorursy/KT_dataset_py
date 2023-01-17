# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import matplotlib.pyplot as plt

from tqdm import tqdm

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')
img_size = 300

def crop_image_from_gray(img,tol=7):

    if img.ndim ==2:

        mask = img>tol

        return img[np.ix_(mask.any(1),mask.any(0))]

    elif img.ndim==3:

        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        mask = gray_img>tol

        

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]

        if (check_shape == 0): # image is too dark so that we crop out everything,

            return img # return original image

        else:

            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]

            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]

            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]

            img = np.stack([img1,img2,img3],axis=-1)

        return img

def preprocess_image(img_file):

    img = cv2.imread(img_file)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = crop_image_from_gray(img)

    img = cv2.resize(img, (img_size,img_size))

    return img
fig=plt.figure(figsize=(8, 8))

image_path = train_df.loc[8,'id_code']

image_id = train_df.loc[8,'diagnosis']

img = preprocess_image(f'../input/train_images/{image_path}.png')

plt.title(f'diagnosis:{image_id} index:{8}')

plt.imshow(img)

plt.tight_layout()
def isClear(img, threshold = 60):

    return cv2.Laplacian(img, cv2.CV_64F).var() > threshold
def display_samples(df, columns=4, rows=3):

    fig=plt.figure(figsize=(5*columns, 4*rows))

    for i in range(columns*rows):

        idx = np.random.randint(0, len(df)-1, 1)[0]

        image_path = df.loc[idx,'id_code']

        image_id = df.loc[idx,'diagnosis']

        img = preprocess_image(f'../input/train_images/{image_path}.png')

        fig.add_subplot(rows, columns, i+1)

        plt.title(f'diagnosis:{image_id}   isclear:{isClear(img)}')

        plt.imshow(img)

    plt.tight_layout()

display_samples(train_df)
import time

blur_list = []

blur_list_id = []

start_time = time.time();

for i, image_id in enumerate(tqdm(train_df['id_code'])):

    img = preprocess_image(f'../input/train_images/{image_id}.png')

    if(not isClear(img)):

        blur_list.append(i)

        blur_list_id.append(image_id)

train_df = train_df.drop(blur_list)

print(f'Cost: {time.time() - start_time}:.3% seconds');
print(f'Droped items:{len(blur_list_id)}')
def display_blurry_samples(df, img_id_list, columns=4, rows=3):

    fig=plt.figure(figsize=(5*columns, 4*rows))

    for i in range(columns*rows):

        img = preprocess_image(f'../input/train_images/{img_id_list[i]}.png')

        fig.add_subplot(rows, columns, i+1)

        plt.title(f'index:{i}  isclear:{isClear(img)}')

        plt.imshow(img)

    plt.tight_layout()

display_blurry_samples(train_df, blur_list_id)