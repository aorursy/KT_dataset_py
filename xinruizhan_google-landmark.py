# this notebook is edied based on https://www.kaggle.com/sandy1112/create-and-train-resnet50-from-scratch

import numpy as np 

import pandas as pd

import os

import tensorflow as tf

import cv2

import skimage.io

from skimage.transform import resize

from imgaug import augmenters as iaa

from sklearn import preprocessing

from sklearn.preprocessing import LabelBinarizer,LabelEncoder

from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras import layers

from tensorflow.keras.layers import Input, Add, Dense, Dropout, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,GlobalAveragePooling2D,Concatenate,concatenate, ReLU, LeakyReLU,Reshape, Lambda

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

from tensorflow.keras.optimizers import Adam,SGD

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential, load_model, Model

from tensorflow.keras.callbacks import LearningRateScheduler

from tensorflow.keras.preprocessing import image

from tensorflow.keras.utils import to_categorical

from tensorflow.keras import metrics

from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications.imagenet_utils import preprocess_input

from tensorflow.keras.initializers import glorot_uniform

from tqdm import tqdm

import imgaug as ia

from imgaug import augmenters as iaa

from PIL import Image

import keras.backend as K

K.set_image_data_format('channels_last')

# 

K.set_learning_phase(1)
os.listdir('/kaggle/input/landmark-retrieval-2020')

train_df = pd.read_csv('/kaggle/input/landmark-retrieval-2020/train.csv')

train_df.head()

dct = train_df.landmark_id.value_counts().to_dict()
rows = []

for row in tqdm(train_df.id):

    path  = list(row)[:3]

    temp = f"../input/landmark-retrieval-2020/train/{path[0]}/{path[1]}/{path[2]}/{row}.jpg"

    rows.append(temp)

    # print(temp)

rows = pd.DataFrame(rows)

rows['landmark_id'] = train_df.landmark_id

rows['count'] = rows.landmark_id.apply(lambda x: dct[x])
target = rows.landmark_id.value_counts().argsort()[:100].index.to_list()

target
top_ten = target[:10]

top_one = top_ten[0]

rows[rows.landmark_id == top_one][0].to_list()[0]
example = cv2.imread('../input/landmark-retrieval-2020/train/0/0/0/0006f34cf361f69c.jpg')

import matplotlib.pyplot as plt

plt.imshow(example)
top_ten[0]
os.makedirs('../working/training', exist_ok=True)
for item in tqdm(top_ten):

    pa = os.path.join('../working/training', str(item))

    os.makedirs(pa, exist_ok=True)

    paths = rows[rows.landmark_id == item][0].to_list()[:500]

    for i in range(len(paths)):

        path = paths[i]

        target_path = os.path.join(pa, str(i)+'.jpg')

        temp = cv2.imread(path)

        cv2.imwrite(target_path, temp)
import zipfile

def zipdir(path, ziph):

    # ziph is zipfile handle

    for root, dirs, files in os.walk(path):

        for file in files:

            ziph.write(os.path.join(root, file))

zipf = zipfile.ZipFile('samples.zip', 'w')# , zipfile.ZIP_DEFLATED

zipdir('../working/training', zipf)

zipf.close()