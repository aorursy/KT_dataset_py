# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from skimage import io
import matplotlib.pyplot as plt
import cv2
from tqdm.notebook import tqdm
base = '/kaggle/input/prostate-cancer-grade-assessment'
train_df = pd.read_csv(f'{base}/train.csv')
test_df = pd.read_csv(f'{base}/test.csv')
sample_submit_df = pd.read_csv(f'{base}/sample_submission.csv')
train_dct = train_df[['image_id', 'isup_grade']].set_index('image_id').to_dict()['isup_grade']
os.listdir(base)
train_images_dir = f'{base}/train_images'
train_label_masks = f'{base}/train_label_masks'
display(train_df.head(), test_df.head())
display(len([file for file in os.listdir(train_images_dir)]), os.listdir(train_images_dir)[:5])
display(len([file for file in os.listdir(train_label_masks)]), os.listdir(train_label_masks)[:5])
# edited based on https://www.kaggle.com/xhlulu/panda-resize-and-save-train-data
first_image_path = os.path.join(train_images_dir, os.listdir(train_images_dir)[0])
first_image_bio = io.MultiImage(first_image_path)
display(first_image_bio[-1].shape, len(first_image_bio))
first_image = cv2.resize(first_image_bio[-1], (512, 512))
plt.imshow(first_image)
# edited based on https://www.kaggle.com/xhlulu/panda-resize-and-save-train-data
first_mask_path = os.path.join(train_label_masks, os.listdir(train_label_masks)[0])
first_mask_bio = io.MultiImage(first_mask_path)
display(first_mask_bio[1].shape, len(first_mask_bio))
first_mask = cv2.resize(first_mask_bio[1], (512, 512))
plt.imshow(first_mask)
# edited based on https://www.kaggle.com/xhlulu/panda-resize-and-save-train-data
image_dir_train = '/kaggle/working/trainning_panda_i/train'
image_dir_validation = '/kaggle/working/trainning_panda_i/validation'
mask_dir = '/kaggle/working/trainning_panda_m'
os.makedirs(image_dir_train, exist_ok=True)
os.makedirs(image_dir_validation, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)
for i in range(6):
    if not os.path.isdir(os.path.join(image_dir_train, str(i))):
        os.mkdir(os.path.join(image_dir_train, str(i)))
    if not os.path.isdir(os.path.join(image_dir_validation, str(i))):
        os.mkdir(os.path.join(image_dir_validation, str(i)))
# edited based on https://www.kaggle.com/xhlulu/panda-resize-and-save-train-data
images = os.listdir(train_images_dir)
images_train = images[:8000]
images_validation = images[8000:]
for image in tqdm(images_train):
    temp = os.path.join(train_images_dir, image)
    id_ = image[:-5]
    label = train_dct[id_]
    img_name = id_ + '.png'
    img_dir = os.path.join(image_dir_train, str(label))
    save_path = os.path.join(img_dir, img_name)
    biopsy = io.MultiImage(temp)
    img = cv2.resize(biopsy[-1], (512, 512))
    cv2.imwrite(save_path, img)
for image in tqdm(images_validation):
    temp = os.path.join(train_images_dir, image)
    id_ = image[:-5]
    label = train_dct[id_]
    img_name = id_ + '.png'
    img_dir = os.path.join(image_dir_validation, str(label))
    save_path = os.path.join(img_dir, img_name)
    biopsy = io.MultiImage(temp)
    img = cv2.resize(biopsy[-1], (512, 512))
    cv2.imwrite(save_path, img)
# edited based on https://www.kaggle.com/xhlulu/panda-resize-and-save-train-data
masks = os.listdir(train_label_masks)
for mask in tqdm(masks):
    temp = os.path.join(train_label_masks, mask)
    save_path = mask_dir + '/'+ mask[:-5] + '.png'
    biopsy = io.MultiImage(temp)
    img = cv2.resize(biopsy[-1], (512, 512))
    cv2.imwrite(save_path, img)

img = cv2.imread(os.path.join(image_dir, os.listdir(image_dir)[0]))
plt.imshow(img)

dic = train_df.to_dict()
def display_mandi(mask):
    pic, (ax1, ax2) = plt.subplots(ncol=2, figsize=(8,8))
    id = mask[:-9]
    ax1.imshow()
os.listdir('/kaggle/trainning_panda_m')[1][:-9]
os.listdir(image_dir)[1][:-4]
os.listdir('/kaggle')
