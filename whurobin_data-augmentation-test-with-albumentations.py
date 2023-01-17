import os
import ast
from collections import namedtuple

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm
from PIL import Image

import joblib
from joblib import Parallel, delayed

import cv2
import albumentations
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.core.transforms_interface import DualTransform
from albumentations.augmentations.bbox_utils import denormalize_bbox, normalize_bbox

from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.image import imsave
# Constants
BASE_DIR = '/kaggle/input/global-wheat-detection'
WORK_DIR = '/kaggle/working'

# Set seed for numpy for reproducibility
np.random.seed(1996)
train_df = pd.read_csv(os.path.join(BASE_DIR, 'train.csv'))

# Let's expand the bounding box coordinates and calculate the area of all the bboxes
train_df[['x_min','y_min', 'width', 'height']] = pd.DataFrame([ast.literal_eval(x) for x in train_df.bbox.tolist()], index= train_df.index)
train_df = train_df[['image_id', 'bbox', 'source', 'x_min', 'y_min', 'width', 'height']]
train_df['area'] = train_df['width'] * train_df['height']
train_df['x_max'] = train_df['x_min'] + train_df['width']
train_df['y_max'] = train_df['y_min'] + train_df['height']
train_df = train_df.drop(['bbox', 'source'], axis=1)
train_df = train_df[['image_id', 'x_min', 'y_min', 'x_max', 'y_max', 'width', 'height', 'area']]

# There are some buggy annonations in training images having huge bounding boxes. Let's remove those bboxes
train_df = train_df[train_df['area'] < 100000]

train_df.head()
print(train_df.shape)
image_ids = train_df['image_id'].unique()
print(f'Total number of training images: {len(image_ids)}')
# Read the image on which data augmentaion is to be performed
image_id = 'c14c1e300'
image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
image /= 255.0
plt.figure(figsize = (10, 10))
plt.imshow(image)
plt.show()
pascal_voc_boxes = train_df[train_df['image_id'] == image_id][['x_min', 'y_min', 'x_max', 'y_max']].astype(np.int32).values
coco_boxes = train_df[train_df['image_id'] == image_id][['x_min', 'y_min', 'width', 'height']].astype(np.int32).values
assert(len(pascal_voc_boxes) == len(coco_boxes))
labels = np.ones((len(pascal_voc_boxes), ))
def get_bbox(bboxes, col, color='white', bbox_format='pascal_voc'):
    
    for i in range(len(bboxes)):
        # Create a Rectangle patch
        if bbox_format == 'pascal_voc':
            rect = patches.Rectangle(
                (bboxes[i][0], bboxes[i][1]),
                bboxes[i][2] - bboxes[i][0], 
                bboxes[i][3] - bboxes[i][1], 
                linewidth=2, 
                edgecolor=color, 
                facecolor='none')
        else:
            rect = patches.Rectangle(
                (bboxes[i][0], bboxes[i][1]),
                bboxes[i][2], 
                bboxes[i][3], 
                linewidth=2, 
                edgecolor=color, 
                facecolor='none')

        # Add the patch to the Axes
        col.add_patch(rect)
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.Blur(p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.VerticalFlip(p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.HorizontalFlip(p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.Flip(p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.Normalize(p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.Transpose(p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.RandomCrop(height=400, width=400, p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.RandomGamma( p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.RandomRotate90( p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.Rotate(limit=30, p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.ShiftScaleRotate(p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.CenterCrop(400, 400, p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.HueSaturationValue(hue_shift_limit=0.5, sat_shift_limit= 0.5, val_shift_limit=0.5, p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.PadIfNeeded(800, 800, p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.RGBShift(r_shift_limit=0.5, g_shift_limit=0.5, b_shift_limit=0.5,p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.RandomBrightness(limit=0.2, p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.RandomContrast(limit=0.2, p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.MotionBlur(p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.MedianBlur(blur_limit=3, p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.GaussianBlur(blur_limit=3, p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.GaussNoise(var_limit=(0.1, 0.1), p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.GlassBlur(sigma=0.1, max_delta=4, p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
# Read the image on which data augmentaion is to be performed
image_id = 'c14c1e300'
image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.CLAHE(p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
# Read the image on which data augmentaion is to be performed
image_id = 'c14c1e300'
image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
image /= 255.0
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.ChannelShuffle(p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
# Read the image on which data augmentaion is to be performed
image_id = 'c14c1e300'
image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
image /= 255.0
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.InvertImg(p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
# Read the image on which data augmentaion is to be performed
image_id = 'c14c1e300'
image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
image /= 255.0
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.ToGray(p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
# Read the image on which data augmentaion is to be performed
image_id = 'c14c1e300'
image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
image /= 255.0
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.ToSepia(p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
# Read the image on which data augmentaion is to be performed
image_id = 'c14c1e300'
image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
image /= 255.0
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.JpegCompression(p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
# Read the image on which data augmentaion is to be performed
image_id = 'c14c1e300'
image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
image /= 255.0
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.ImageCompression(p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
# Read the image on which data augmentaion is to be performed
image_id = 'c14c1e300'
image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
image /= 255.0
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
# Read the image on which data augmentaion is to be performed
image_id = 'c14c1e300'
image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
image /= 255.0
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.CoarseDropout(max_holes=8, max_height=64, max_width=64,p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
# Read the image on which data augmentaion is to be performed
image_id = 'c14c1e300'
image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.CLAHE(p=1),
        albumentations.ToFloat(p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image/255.0)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
# Read the image on which data augmentaion is to be performed
image_id = 'c14c1e300'
image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
image /= 255.0
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.FromFloat(dtype='uint8', p=1),
        albumentations.CLAHE(p=1),
        albumentations.ToFloat(p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
# Read the image on which data augmentaion is to be performed
image_id = 'c14c1e300'
image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
image /= 255.0
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.Crop( x_max=400, y_max=400,p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
# Read the image on which data augmentaion is to be performed
image_id = 'c14c1e300'
image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
image /= 255.0
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.RandomScale(scale_limit=0.3, p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
# Read the image on which data augmentaion is to be performed
image_id = 'c14c1e300'
image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
image /= 255.0
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.LongestMaxSize(400, p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
# Read the image on which data augmentaion is to be performed
image_id = 'c14c1e300'
image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
image /= 255.0
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.RandomSizedCrop(min_max_height=(400, 400), height=512, width=512, p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
# Read the image on which data augmentaion is to be performed
image_id = 'c14c1e300'
image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
image /= 255.0
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.RandomResizedCrop(height=512, width=512, p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
class CustomCutout(DualTransform):
    """
    Custom Cutout augmentation with handling of bounding boxes 
    Note: (only supports square cutout regions)
    
    Author: Kaushal28
    Reference: https://arxiv.org/pdf/1708.04552.pdf
    """
    
    def __init__(
        self,
        fill_value=0,
        bbox_removal_threshold=0.50,
        min_cutout_size=192,
        max_cutout_size=512,
        number=1,
        always_apply=False,
        p=0.5
    ):
        """
        Class construstor
        :param fill_value: Value to be filled in cutout (default is 0 or black color)
        :param bbox_removal_threshold: Bboxes having content cut by cutout path more than this threshold will be removed
        :param min_cutout_size: minimum size of cutout (192 x 192)
        :param max_cutout_size: maximum size of cutout (512 x 512)
        """
        super(CustomCutout, self).__init__(always_apply, p)  # Initialize parent class
        self.fill_value = fill_value
        self.bbox_removal_threshold = bbox_removal_threshold
        self.min_cutout_size = min_cutout_size
        self.max_cutout_size = max_cutout_size
        self.number = number
        
    def _get_cutout_position(self, img_height, img_width, cutout_size):
        """
        Randomly generates cutout position as a named tuple
        
        :param img_height: height of the original image
        :param img_width: width of the original image
        :param cutout_size: size of the cutout patch (square)
        :returns position of cutout patch as a named tuple
        """
        position = namedtuple('Point', 'x y')
        return position(
            np.random.randint(0, img_width - cutout_size + 1),
            np.random.randint(0, img_height - cutout_size + 1)
        )
    def _get_cutout(self, img_height, img_width):
        """
        Creates a cutout pacth with given fill value and determines the position in the original image
        
        :param img_height: height of the original image
        :param img_width: width of the original image
        :returns (cutout patch, cutout size, cutout position)
        """
        cutout_size = np.random.randint(self.min_cutout_size, self.max_cutout_size + 1)
        cutout_position = self._get_cutout_position(img_height, img_width, cutout_size)
        return np.full((cutout_size, cutout_size, 3), self.fill_value), cutout_size, cutout_position
    def apply(self, image, **params):
        """
        Applies the cutout augmentation on the given image
        
        :param image: The image to be augmented
        :returns augmented image
        """
        image = image.copy()  # Don't change the original image
        self.img_height, self.img_width, _ = image.shape
        for i in range(self.number):
            cutout_arr, cutout_size, cutout_pos = self._get_cutout(self.img_height, self.img_width)
            
            # Set to instance variables to use this later
            self.image = image
            self.cutout_pos = cutout_pos
            self.cutout_size = cutout_size
            
            image[cutout_pos.y:cutout_pos.y+cutout_size, cutout_pos.x:cutout_size+cutout_pos.x, :] = cutout_arr
        return image
    def apply_to_bbox(self, bbox, **params):
        """
        Removes the bounding boxes which are covered by the applied cutout
        
        :param bbox: A single bounding box coordinates in pascal_voc format
        :returns transformed bbox's coordinates
        """

        # Denormalize the bbox coordinates
        bbox = denormalize_bbox(bbox, self.img_height, self.img_width)
        x_min, y_min, x_max, y_max = tuple(map(int, bbox))
        if x_min >= x_max or y_min >= y_max:
            return normalize_bbox((0, 0, 0, 0), self.img_height, self.img_width)

        bbox_size = (x_max - x_min) * (y_max - y_min)  # width * height
        overlapping_size = np.sum(
            (self.image[y_min:y_max, x_min:x_max, 0] == self.fill_value) &
            (self.image[y_min:y_max, x_min:x_max, 1] == self.fill_value) &
            (self.image[y_min:y_max, x_min:x_max, 2] == self.fill_value)
        )
        # Remove the bbox if it has more than some threshold of content is inside the cutout patch
        if overlapping_size / bbox_size > self.bbox_removal_threshold:
            return normalize_bbox((0, 0, 0, 0), self.img_height, self.img_width)

        return normalize_bbox(bbox, self.img_height, self.img_width)

    def get_transform_init_args_names(self):
        """
        Fetches the parameter(s) of __init__ method
        :returns: tuple of parameter(s) of __init__ method
        """
        return ('fill_value', 'bbox_removal_threshold', 'min_cutout_size', 'max_cutout_size', 'always_apply', 'p')
        
# Read the image on which data augmentaion is to be performed
image_id = 'c14c1e300'
image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
image /= 255.0
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        CustomCutout(bbox_removal_threshold=0.50,min_cutout_size=32,max_cutout_size=96,number=12,p=1),
#         albumentations.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.8, p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
# Read the image on which data augmentaion is to be performed
image_id = 'c14c1e300'
image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
image /= 255.0
aug = albumentations.Compose([
#         albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
         albumentations.PadIfNeeded(1200, 1200, p=1),
         albumentations.RandomSizedBBoxSafeCrop(height=1024, width=1024, erosion_rate=4, p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
# Read the image on which data augmentaion is to be performed
image_id = 'c14c1e300'
image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
image /= 255.0
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.RandomSnow(p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
# Read the image on which data augmentaion is to be performed
image_id = 'c14c1e300'
image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
image /= 255.0
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.RandomRain(p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
# Read the image on which data augmentaion is to be performed
image_id = 'c14c1e300'
image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
image /= 255.0
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.RandomFog(p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
# Read the image on which data augmentaion is to be performed
image_id = 'c14c1e300'
image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
image /= 255.0
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), src_radius=200, src_color=(255, 255, 255), p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
# Read the image on which data augmentaion is to be performed
image_id = 'c14c1e300'
image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
image /= 255.0
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.RandomShadow( p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
# Read the image on which data augmentaion is to be performed
image_id = 'c14c1e300'
image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
image /= 255.0
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.ChannelDropout( p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
# Read the image on which data augmentaion is to be performed
image_id = 'c14c1e300'
image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# .astype(np.float32)
# image /= 255.0
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.ISONoise( p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
# Read the image on which data augmentaion is to be performed
image_id = 'c14c1e300'
image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# .astype(np.float32)
# image /= 255.0
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.Solarize(threshold=224, p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
# Read the image on which data augmentaion is to be performed
image_id = 'c14c1e300'
image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# .astype(np.float32)
# image /= 255.0
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.Equalize( p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
# Read the image on which data augmentaion is to be performed
image_id = 'c14c1e300'
image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# .astype(np.float32)
# image /= 255.0
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.Posterize( p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
# Read the image on which data augmentaion is to be performed
image_id = 'c14c1e300'
image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
image /= 255.0
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.Downscale( p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
# Read the image on which data augmentaion is to be performed
image_id = 'c14c1e300'
image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
image /= 255.0
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.MultiplicativeNoise( p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
# Read the image on which data augmentaion is to be performed
image_id = 'c14c1e300'
image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
image /= 255.0
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.DualIAATransform( p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
# Read the image on which data augmentaion is to be performed
image_id = 'c14c1e300'
image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
image /= 255.0
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.ImageOnlyIAATransform( p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
# Read the image on which data augmentaion is to be performed
image_id = 'c14c1e300'
image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
image /= 255.0
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.IAAEmboss( p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
# Read the image on which data augmentaion is to be performed
image_id = 'c14c1e300'
image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
image /= 255.0
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.IAASuperpixels( p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
# Read the image on which data augmentaion is to be performed
image_id = 'c14c1e300'
image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
image /= 255.0
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.IAASharpen( p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
# Read the image on which data augmentaion is to be performed
image_id = 'c14c1e300'
image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
image /= 255.0
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.IAAAdditiveGaussianNoise( p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
# Read the image on which data augmentaion is to be performed
image_id = 'c14c1e300'
image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
image /= 255.0
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.IAACropAndPad( p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
# Read the image on which data augmentaion is to be performed
image_id = 'c14c1e300'
image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
image /= 255.0
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.IAAFliplr( p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
# Read the image on which data augmentaion is to be performed
image_id = 'c14c1e300'
image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
image /= 255.0
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.IAAFlipud( p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
# Read the image on which data augmentaion is to be performed
image_id = 'c14c1e300'
image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
image /= 255.0
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.IAAAffine(scale=1.0, rotate=10, shear=5., order=1, cval=0, p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()
# Read the image on which data augmentaion is to be performed
image_id = 'c14c1e300'
image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
image /= 255.0
aug = albumentations.Compose([
        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512
#         albumentations.VerticalFlip(1),    # Verticlly flip the image
        albumentations.IAAPiecewiseAffine(scale=(0.03, 0.05), nb_rows=4, nb_cols=4, order=1, cval=0,  p=1),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
get_bbox(pascal_voc_boxes, ax[0], color='red')
ax[0].title.set_text('Original Image')
ax[0].imshow(image)

get_bbox(aug_result['bboxes'], ax[1], color='red')
ax[1].title.set_text('Augmented Image')
ax[1].imshow(aug_result['image'])
plt.show()