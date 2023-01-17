# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from tensorflow import keras

import cv2

import albumentations

import matplotlib.pyplot as plt

import seaborn as sns

import random



print(tf.__version__)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames[:5]:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
CLASSS = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative DR"}

SEED = 77

random.seed(SEED)

IMG_CHANNELS = 3

IMG_WIDTH = 512



# These are used for histogram equalization

clipLimit=2.0 

tileGridSize=(8, 8)  



channels = {"R":0, "G": 1, "B":2}
sample_submission = pd.read_csv("/kaggle/input/aptos2019-blindness-detection/sample_submission.csv")

print(sample_submission.head())

test_file = pd.read_csv("/kaggle/input/aptos2019-blindness-detection/test.csv")

train_file = pd.read_csv("/kaggle/input/aptos2019-blindness-detection/train.csv")

print(test_file.head())

print(train_file.head())
# Now check the distribution of train images

print(len(train_file))

train_file['diagnosis'].hist(figsize = (8,4))
def display_samples(df, columns=4, rows=3):

    fig=plt.figure(figsize=(4*columns, 3*rows))

    

    random_indices = random.sample(range(0, len(train_file)), columns*rows)

    count = 0

    for i in random_indices:

        image_path = df.loc[i,'id_code']

        image_rating = df.loc[i,'diagnosis']

        img = cv2.imread(f'../input/aptos2019-blindness-detection/train_images/{image_path}.png')

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        

        fig.add_subplot(rows, columns, count+1)

        count += 1

        plt.title(image_rating)

        plt.imshow(img)

    

    plt.tight_layout()



display_samples(train_file, 4, 8)
sample_img_path = random.choice(train_file["id_code"])

sample_img = cv2.imread(f'../input/aptos2019-blindness-detection/train_images/{sample_img_path}.png')

sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)

print(sample_img.shape)

plt.imshow(sample_img)
def draw_img(imgs, class_label='0'):

    fig, axis = plt.subplots(2, 6, figsize=(15, 6))

    for idnx, (idx, row) in enumerate(imgs.iterrows()):

        imgPath = (f'../input/aptos2019-blindness-detection/train_images/{row["id_code"]}.png')

        img = cv2.imread(imgPath)

        row = idnx // 6

        col = idnx % 6

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        axis[row, col].imshow(img)

    plt.suptitle(class_label)

    plt.show()
CLASS_ID = 0

draw_img(train_file[train_file.diagnosis == CLASS_ID].head(12), CLASSS[CLASS_ID])
CLASS_ID = 4

draw_img(train_file[train_file.diagnosis == CLASS_ID].head(12), CLASSS[CLASS_ID])
# HE --> Histogram Equalization: True to apply CLAHE to the color channel image



print(channels)

print(clipLimit)

print(tileGridSize)



def display_single_channel_samples(df, columns=4, rows=3, channel = "G", HE = False):

    fig=plt.figure(figsize=(4*columns, 3*rows))

    random.seed(SEED) # This lines make sure that all the following function calls will

                    # show the same set of randomly selected images

    random_indices = random.sample(range(0, len(train_file)), columns*rows)

    

    count = 0

    for i in random_indices:

        # Load images and convert to RGB

        image_path = df.loc[i,'id_code']

        image_rating = df.loc[i,'diagnosis']

        img = cv2.imread(f'../input/aptos2019-blindness-detection/train_images/{image_path}.png')

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        

        # Apply some pre-processing

        img = img[:,:,channels[channel]]

        if HE: #If the histogram equalization is applied

            clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)

            img = clahe.apply(img) #This is for creating the image with a higher contrast

        else:

            pass

        

        # Actually drawing stuff 

        fig.add_subplot(rows, columns, count+1)

#         fig.add_subplot()

        count += 1

        plt.title(image_rating)

        plt.imshow(img)

    

    plt.tight_layout()

display_single_channel_samples(train_file, 4, 3)
display_single_channel_samples(train_file,4,3, "G", HE = True)
display_single_channel_samples(train_file,4,3, "R", HE = True)
display_single_channel_samples(train_file,4,2, "B", HE = False)
display_single_channel_samples(train_file,4,2, "B", HE = True)
def resize_bens(df, columns=4, rows=3, sigmaX = 20, img_width = IMG_WIDTH): # Assume image is square 

    fig=plt.figure(figsize=(4*columns, 3*rows))

    

    random.seed(SEED)

    random_indices = random.sample(range(0, len(train_file)), columns*rows)

    count = 0

    for i in random_indices:

        image_path = df.loc[i,'id_code']

        image_rating = df.loc[i,'diagnosis']

        img = cv2.imread(f'../input/aptos2019-blindness-detection/train_images/{image_path}.png')

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (img_width, img_width))

        img = cv2.addWeighted ( img,4, cv2.GaussianBlur(img , (0,0) , sigmaX) ,-4 ,128)

        

        fig.add_subplot(rows, columns, count+1)

        count += 1

        plt.title(image_rating)

        plt.imshow(img)

    

    plt.tight_layout()
resize_bens(train_file, 4, 4)
resize_bens(train_file,4,4, sigmaX = 50)
resize_bens(train_file,4,4, sigmaX = 16)
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

    #         print(img1.shape,img2.shape,img3.shape)

            img = np.stack([img1,img2,img3],axis=-1)

    #         print(img.shape)

        return img
def resize_bens_and_crop(df, columns=4, rows=3, sigmaX = 20, img_width = IMG_WIDTH): # Assume image is square 

    fig=plt.figure(figsize=(4*columns, 3*rows))

    

    random.seed(SEED)

    random_indices = random.sample(range(0, len(train_file)), columns*rows)

    count = 0

    for i in random_indices:

        image_path = df.loc[i,'id_code']

        image_rating = df.loc[i,'diagnosis']

        img = cv2.imread(f'../input/aptos2019-blindness-detection/train_images/{image_path}.png')

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        

        # First crop, then resize.

        img = crop_image_from_gray(img)

        img = cv2.resize(img, (img_width, img_width))

        

        # Applying Ben's method

        img = cv2.addWeighted ( img,4, cv2.GaussianBlur(img , (0,0) , sigmaX) ,-4 ,128)

        

        fig.add_subplot(rows, columns, count+1)

        count += 1

        plt.title(image_rating)

        plt.imshow(img)

    

    plt.tight_layout()
resize_bens_and_crop(train_file, 4, 4, sigmaX = 10)
# HE --> Histogram Equalization: Try to apply CLAHE to the color channel image



print(channels)

print(clipLimit)

print(tileGridSize)



def display_single_channel_crop_resize(df, columns=4, rows=3, channel = "G", HE = False):

    fig=plt.figure(figsize=(4*columns, 3*rows))

    random.seed(SEED) # This lines make sure that all the following function calls will

                    # show the same set of randomly selected images

    random_indices = random.sample(range(0, len(train_file)), columns*rows)

    

    count = 0

    for i in random_indices:

        # Load images and convert to RGB

        image_path = df.loc[i,'id_code']

        image_rating = df.loc[i,'diagnosis']

        img = cv2.imread(f'../input/aptos2019-blindness-detection/train_images/{image_path}.png')

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        

        # Crop and then resize the image

        img = crop_image_from_gray(img)

        img = cv2.resize(img, (IMG_WIDTH, IMG_WIDTH))

        

        # Apply some pre-processing

        img = img[:,:,channels[channel]]

        if HE: #If the histogram equalization is applied

            clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)

            img = clahe.apply(img) #This is for creating the image with a higher contrast

        else:

            pass

        

        # Actually drawing stuff 

        fig.add_subplot(rows, columns, count+1)

#         fig.add_subplot()

        count += 1

        plt.title(image_rating)

        plt.imshow(img)

    

    plt.tight_layout()

resize_bens_and_crop(train_file, 5, 7, sigmaX = 10)
display_single_channel_crop_resize(train_file, 5, 7, HE = True)