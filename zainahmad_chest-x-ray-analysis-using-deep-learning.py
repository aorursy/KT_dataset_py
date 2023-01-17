import pandas as pd

import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt

import seaborn as sns

import os

from PIL import Image

from tensorflow.keras.layers import GlobalAveragePooling2D ,Input, Flatten , Dense , Dropout , Concatenate ,Activation

from tensorflow.keras.models import Model , Sequential

from tensorflow.keras.applications import mobilenet_v2

import shutil

from glob import glob
data_entry = pd.read_csv('../input/data/Data_Entry_2017.csv')

bbox_list = pd.read_csv('../input/data/BBox_List_2017.csv')

all_image_paths = {os.path.basename(x): x for x in 

                   glob(os.path.join('..', 'input','data', 'images*','*','*.png'))}



data_entry.head()
data_entry.shape
bbox_list.head()
bbox_list.shape
fig , ax  = plt.subplots(figsize=(12,6))

sns.countplot(bbox_list['Finding Label'] , ax=ax)

ax.set_title('Class Distribution')

plt.show()
bbox_list['Finding Label'].value_counts()
data_entry['Finding Labels'].nunique()
len(data_entry[data_entry['Finding Labels'] == 'No Finding'])
df_one_hot = data_entry.copy()

for i, v in enumerate(df_one_hot['Finding Labels'].values):

    v = v.split('|')

    for c in v:

        df_one_hot.loc[i , c] = 1

df_one_hot.head()
df_one_hot.columns
df = df_one_hot[['Image Index', 'Cardiomegaly', 'Emphysema', 'Effusion', 'No Finding', 'Hernia',

       'Infiltration', 'Mass', 'Nodule', 'Atelectasis', 'Pneumothorax',

       'Pleural_Thickening', 'Pneumonia', 'Fibrosis', 'Edema',

       'Consolidation']]
df = df.fillna(0)

df.head()
for c in df.columns[1:]:

    df[c] = df[c].astype(int)

    

df.head()
class_dist = { i : df[i].sum() for i in df.columns[1:]}

cl = list(class_dist.keys())

freq = list(class_dist.values())

fig , ax = plt.subplots(figsize=(12,6))

sns.barplot(x=freq , y = cl , ax=ax)

ax.set_title('Class Distibtuion')

fig.show()
class_dist
base_model = mobilenet_v2.MobileNetV2(include_top=False ,weights='imagenet')

pre_process = mobilenet_v2.preprocess_input
df_inf = df.loc[df['Infiltration'] == 1]

print(df_inf.shape)

df_inf.head()
fig = plt.figure(figsize=(15, 15))

i = 1

for ii in df_inf['Image Index'].values[:9]:

    img = plt.imread(all_image_paths[ii])

    fig.add_subplot(3,3,i)

    plt.imshow(img , cmap='Greys_r')

    i+=1

    

plt.title('Some examples of infiltration')

fig.tight_layout(pad=2)

fig.show()