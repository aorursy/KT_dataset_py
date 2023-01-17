import os
import sys
import math

from collections import defaultdict
from multiprocessing.dummy import Pool as ThreadPool

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
%matplotlib inline

scale = 1.5
plt.rcParams['figure.figsize'] = [6.4*scale, 4.8*scale]
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
os.listdir('../input')
len(list(open('../input/train_ship_segmentations.csv'))) - 1
len(list(open('../input/sample_submission.csv'))) - 1
train_path = '../input/train'
test_path = '../input/test'
train_files = os.listdir(f'{train_path}')
print(len(train_files))

test_files = os.listdir(f'{test_path}')
print(len(test_files))
im = plt.imread(f'{test_path}/fec9bf8f4.jpg')
plt.imshow(im)
im.shape
idx = np.random.permutation(len(train_files))[:9]

fig = plt.figure(figsize=(10, 10))
fig.subplots_adjust(wspace=0, hspace=0)
for i, id in enumerate(idx):
    fig.add_subplot(3, 3, i + 1)

    im = plt.imread(f'{train_path}/{train_files[id]}')
    plt.imshow(im)
    plt.axis('off')

plt.show()
def get_im_shape(fpath):
    im = plt.imread(fpath)
    return im.shape

pool = ThreadPool(4)

train_fpaths = [os.path.join(train_path, fname) for fname in train_files]
train_im_shapes = pool.map(get_im_shape, train_fpaths)

test_fpaths = [os.path.join(test_path, fname) for fname in test_files]
test_im_shapes = pool.map(get_im_shape, test_fpaths)
counter = defaultdict(int)
for shape in train_im_shapes:
    counter[len(shape)] += 1
print(f'All train images have a channel: {counter}')
# There is one invalid image
invalid_idx = [i for i in range(len(train_im_shapes)) if len(train_im_shapes[i]) != 3][0]
invalid_idx
os.path.isfile(train_fpaths[invalid_idx])
im = plt.imread(train_fpaths[invalid_idx])
im.shape
print(f'Don\'t use image: {train_fpaths[invalid_idx]}')
counter = defaultdict(int)
for i, shape in enumerate(train_im_shapes):
    if i == invalid_idx: continue
    counter[shape[2]] += 1
print(f'All train images have 3 channels: {counter}')

counter = defaultdict(int)
for i, shape in enumerate(train_im_shapes):
    if i == invalid_idx: continue
    counter[shape[1]] += 1
print(f'Train images\' width: {counter}')

counter = defaultdict(int)
for i, shape in enumerate(train_im_shapes):
    if i == invalid_idx: continue
    counter[shape[0]] += 1
print(f'Train images\' height: {counter}')
counter = defaultdict(int)
for shape in test_im_shapes:
    counter[len(shape)] += 1
print(f'All test images have a channel: {counter}')

counter = defaultdict(int)
for i, shape in enumerate(test_im_shapes):
    counter[shape[2]] += 1
print(f'All test images have 3 channels: {counter}')

counter = defaultdict(int)
for i, shape in enumerate(test_im_shapes):
    counter[shape[1]] += 1
print(f'Test images\' width: {counter}')

counter = defaultdict(int)
for i, shape in enumerate(test_im_shapes):
    counter[shape[0]] += 1
print(f'Test images\' height: {counter}')
masks = pd.read_csv('../input/train_ship_segmentations.csv')
print(masks.shape)
masks.head()
# How many image ids?
len(masks['ImageId'].unique())
# For images without ship, there is only one line per image id
df_tmp = masks[masks['EncodedPixels'].isna()]

print(len(df_tmp['ImageId'].unique()))
print(len(df_tmp))
# Number of images with ships and without ships
n_im_no_ships = len(masks[masks['EncodedPixels'].isna()]['ImageId'].unique())
n_im_ships = len(masks[~masks['EncodedPixels'].isna()]['ImageId'].unique())
sns.barplot(x=['Ships', 'No ships'], y=[n_im_ships, n_im_no_ships])
# Distribution of number of ships in images
df_tmp = masks[~masks['EncodedPixels'].isna()]
sns.distplot(df_tmp['ImageId'].value_counts().values, kde=False)
# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    im = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        im[lo:hi] = 1
    return im.reshape(shape).T

def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
# One image can have multiple masks so there are multiple rows for the image.
# Use pandas to find all rows with the same image and put their masks together.
#fname = masks[~masks['EncodedPixels'].isna()].sample(1)['ImageId'].values[0]
fname = 'a09398d99.jpg'
im = plt.imread(f'{train_path}/{fname}')
rles = masks.loc[masks['ImageId'] == fname, 'EncodedPixels'].tolist()

all_masks = np.zeros((768, 768))
first_masks = np.zeros((768, 768))
for i, rle in enumerate(rles):
    if i == 0: first_masks += rle_decode(rle)
    all_masks += rle_decode(rle)

fig, axarr = plt.subplots(1, 3)
axarr[0].axis('off')
axarr[1].axis('off')
axarr[2].axis('off')
axarr[0].imshow(im)
axarr[1].imshow(all_masks)
axarr[2].imshow(im)
axarr[2].imshow(all_masks, alpha=0.4)
plt.tight_layout(h_pad=0.1, w_pad=0.1)
plt.show()
rle_encode(first_masks)
rles[0] == rle_encode(first_masks)
fpath = f'{test_path}/fec9bf8f4.jpg'
im = cv2.imread(fpath)

# (centered x, centered y, width, height, rotation in degree, confidence score)
locs = [(305.589397186, 357.82121801, 167.674564232, 31.5170499716, -8.00823881288, 0.999999880791)]
mask = np.zeros(shape=im.shape[0:2])

for loc in locs:

    x, y, w, h, d = loc[0:5]

    theta = np.radians(d)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)

    pts = [(w/2, h/2), (-w/2, h/2), (-w/2, -h/2), (w/2, -h/2)]
    pts = [(p[0] * cos_theta + p[1] * sin_theta,
           -(p[0] * sin_theta) + p[1] * cos_theta) for p in pts]
    pts = [(p[0] + x, p[1] + y) for p in pts]
    pts = [(int(p[0]), int(p[1])) for p in pts]
    pts = np.array(pts)

    im = cv2.fillPoly(im, pts=[pts], color=(255, 0, 0))
    mask = cv2.fillPoly(mask, pts=[np.array(pts)], color=(255, 255, 255))

plt.imshow(im[:, :, (2, 1, 0)])
plt.show()
plt.imshow(mask)
fname = 'a09398d99.jpg'
rles = masks.loc[masks['ImageId'] == fname, 'EncodedPixels'].tolist()
im_mask = rle_decode(rles[1])
plt.imshow(im_mask)
# https://stackoverflow.com/questions/49957431/findcontours-of-a-single-channel-image-in-opencv-python
_, contours, hierarchy = cv2.findContours(im_mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cv2.minAreaRect(contours[0])
# https://www.kaggle.com/raresbarbantan/f2-metric/notebook
# https://www.kaggle.com/sgalwan/airbus-ship-detection-challenge-eda-metrics/notebook
def read_masks(masks, im_name):
    mask_list = masks.loc[masks['ImageId'] == im_name, 'EncodedPixels'].tolist()
    all_masks = np.zeros((len(mask_list), 768, 768))
    for idx, mask in enumerate(mask_list):
        if isinstance(mask, str):
            all_masks[idx] = rle_decode(mask)
    return all_masks

def read_flat_mask(masks, im_name):
    all_masks = read_masks(masks, im_name)
    return np.sum(all_masks, axis=0)

def iou(mask1, mask2):
    i = np.sum((mask1 >= 0.5) & (mask2 >= 0.5))
    u = np.sum((mask1 >= 0.5) | (mask2 >= 0.5))
    return i / (1e-8 + u)
im_name_with_ships = '00021ddc3.jpg'
im_name_with_no_ships = '00003e153.jpg'

im_with_ships = plt.imread(f'{train_path}/00021ddc3.jpg')
im_with_no_ships = plt.imread(f'{train_path}/00003e153.jpg')

_, axarr = plt.subplots(1, 2)
axarr[0].axis('off')
axarr[1].axis('off')
axarr[0].imshow(im_with_ships)
axarr[0].imshow(read_flat_mask(masks, im_name_with_ships), alpha=0.6)
axarr[1].imshow(im_with_no_ships)
axarr[1].imshow(read_flat_mask(masks, im_name_with_no_ships), alpha=0.6)
m = read_flat_mask(masks, im_name_with_ships)
print(f'{iou(m, m)}, {iou(m, np.zeros((768, 768)))}, {iou(m, np.ones((768, 768)))}')

m = read_flat_mask(masks, im_name_with_no_ships)
print(f'{iou(m, m)}, {iou(m, np.zeros((768, 768)))}, {iou(m, np.ones((768, 768)))}')
def f2(true_masks, pred_masks):
    # a correct prediction on no ships in image would have F2 of zero (according to formula),
    # but should be rewarded as 1
    if np.sum(true_masks) == np.sum(pred_masks) == 0:
        return 1.0

    pred_masks = [m for m in pred_masks if np.any(m >= 0.5)]
    true_masks = [m for m in true_masks if np.any(m >= 0.5)]

    f2_total = 0
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    for threshold in thresholds:
        if len(true_masks) == 0:
            tp, fn, fp = 0.0, 0.0, float(len(pred_masks))
        else:
            pred_hits = np.zeros(len(pred_masks), dtype=np.bool)
            true_hits = np.zeros(len(true_masks), dtype=np.bool)

            for i, pred_mask in enumerate(pred_masks):
                for j, true_mask in enumerate(true_masks):
                    if iou(pred_mask, true_mask) > threshold:
                        pred_hits[i] = True
                        true_hits[j] = True

            tp = np.sum(pred_hits)
            fp = len(pred_masks) - tp
            fn = len(true_masks) - np.sum(true_hits)

        f2 = (5*tp)/(5*tp + 4*fn + fp)
        f2_total += f2

    return f2_total / len(thresholds)
m = read_masks(masks, im_name_with_ships)
print(f'{f2(m, m)}, {f2(m, np.zeros((768, 768)))}, {f2(m, np.ones((768, 768)))}')

m = read_masks(masks, im_name_with_no_ships)
print(f'{f2(m, m)}, {f2(m, np.zeros((768, 768)))}, {f2(m, np.ones((768, 768)))}')
# Compute the average F2 on a subset of images with a single blank prediction, images with no ships would get 1 and with ships would get 0. F2 score would be close the ratio of number of images with on ships and number of total images (0.72).
subset_images = 2000
random_files = masks['ImageId'].unique()
np.random.shuffle(random_files)

f2_sum = 0
for fname in random_files[:subset_images]:
    mask = read_masks(masks, fname)
    score = f2(mask, [np.zeros((768, 768))])
    f2_sum += score

print(f2_sum/subset_images)
len(masks[masks['EncodedPixels'].isna()]['ImageId'].unique()) / len(masks['ImageId'].unique())
# https://www.kaggle.com/ezietsman/airbus-eda/notebook
sample = masks[~masks.EncodedPixels.isna()].sample(9)
fig, ax = plt.subplots(3, 3, figsize=(10, 10))
fig.subplots_adjust(wspace=0, hspace=0)
for i, im_id in enumerate(sample.ImageId):
    row, col = i // 3, i % 3

    im = plt.imread(f'{train_path}/{im_id}')
    ax[row, col].imshow(im)
    ax[row, col].axis('off')

plt.show()
ignore_files = ['13703f040.jpg', '14715c06d.jpg', '33e0ff2d5.jpg', '4d4e09f2a.jpg', '877691df8.jpg', '8b909bb20.jpg', 'a8d99130e.jpg', 'ad55c3143.jpg', 'c8260c541.jpg', 'd6c7f17c7.jpg', 'dc3e7c901.jpg', 'e44dffe88.jpg', 'ef87bad36.jpg', 'f083256d8.jpg']

fig = plt.figure(figsize=(10, 17))
fig.subplots_adjust(wspace=0, hspace=0)
for i in range(len(ignore_files)):
    fig.add_subplot(5, 3, i + 1)

    im = plt.imread(f'{test_path}/{ignore_files[i]}')
    plt.imshow(im)
    plt.axis('off')

plt.show()
