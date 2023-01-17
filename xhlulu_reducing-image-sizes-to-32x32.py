import os

import cv2

import math



import numpy as np # linear algebra

from PIL import Image

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
label_df = pd.read_csv('../input/train.csv')

submission_df = pd.read_csv('../input/sample_submission.csv')

label_df.head()
label_df['category_id'].value_counts()[1:16].plot(kind='bar')
def display_samples(df, columns=4, rows=3):

    fig=plt.figure(figsize=(5*columns, 3*rows))



    for i in range(columns*rows):

        image_path = df.loc[i,'file_name']

        image_id = df.loc[i,'category_id']

        img = cv2.imread(f'../input/train_images/{image_path}')

        fig.add_subplot(rows, columns, i+1)

        plt.title(image_id)

        plt.imshow(img)



display_samples(label_df)
def get_pad_width(im, new_shape, is_rgb=True):

    pad_diff = new_shape - im.shape[0], new_shape - im.shape[1]

    t, b = math.floor(pad_diff[0]/2), math.ceil(pad_diff[0]/2)

    l, r = math.floor(pad_diff[1]/2), math.ceil(pad_diff[1]/2)

    if is_rgb:

        pad_width = ((t,b), (l,r), (0, 0))

    else:

        pad_width = ((t,b), (l,r))

    return pad_width



def pad_and_resize(image_path, dataset, pad=False, desired_size=32):

    img = cv2.imread(f'../input/{dataset}_images/{image_path}.jpg')

    

    if pad:

        pad_width = get_pad_width(img, max(img.shape))

        padded = np.pad(img, pad_width=pad_width, mode='constant', constant_values=0)

    else:

        padded = img

    

    resized = cv2.resize(padded, (desired_size,)*2).astype('uint8')

    

    return resized
%%time

train_resized_imgs = []

test_resized_imgs = []



for image_id in label_df['id']:

    train_resized_imgs.append(

        pad_and_resize(image_id, 'train')

    )



for image_id in submission_df['Id']:

    test_resized_imgs.append(

        pad_and_resize(image_id, 'test')

    )
X_train = np.stack(train_resized_imgs)

X_test = np.stack(test_resized_imgs)



target_dummies = pd.get_dummies(label_df['category_id'])

train_label = target_dummies.columns.values

y_train = target_dummies.values



print(X_train.shape)

print(X_test.shape)

print(y_train.shape)
# No need to save the IDs of X_test, since they are in the same order as the 

# ID column in sample_submission.csv

np.save('X_train.npy', X_train)

np.save('X_test.npy', X_test)

np.save('y_train.npy', y_train)