# linear algebra

import numpy as np

# data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd

#Unix commands

import os



# import useful tools

from glob import glob

from PIL import Image

import cv2

import pydicom

import scipy.ndimage

from skimage import measure 

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from skimage.morphology import disk, opening, closing

from tqdm import tqdm

from os import listdir, mkdir



from IPython.display import HTML

from PIL import Image



# import data visualization

import matplotlib.pyplot as plt

import matplotlib.patches as patches

import seaborn as sns

import plotly.express as px

from random import randint



from bokeh.plotting import figure

from bokeh.io import output_notebook, show, output_file

from bokeh.models import ColumnDataSource, HoverTool, Panel

from bokeh.models.widgets import Tabs



# import data augmentation

import albumentations as albu



# import math module

import math



#Libraries

import pandas_profiling

import xgboost as xgb

from sklearn.metrics import log_loss

from sklearn.preprocessing import LabelEncoder

from sklearn import preprocessing

from sklearn.model_selection import KFold

from sklearn.tree import DecisionTreeRegressor



#used for changing color of text in print statement

from colorama import Fore, Back, Style

y_ = Fore.YELLOW

r_ = Fore.RED

g_ = Fore.GREEN

b_ = Fore.BLUE

m_ = Fore.MAGENTA

sr_ = Style.RESET_ALL



# One-hot encoding

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestRegressor
# Setup the paths to train and test images

DATASET = '../input/competitive-data-science-predict-future-sales'

TEST_PATH = '../input/competitive-data-science-predict-future-sales/test.csv'

TRAIN_PATH = '../input/competitive-data-science-predict-future-sales/sales_train.csv'

ITEM_PATH = '../input/competitive-data-science-predict-future-sales/items.csv'

CATEGORY_PATH = '../input/competitive-data-science-predict-future-sales/item_categories.csv'

SHOP_PATH = '../input/competitive-data-science-predict-future-sales/shops.csv'
# Loading training data and test data

test = pd.read_csv(TEST_PATH)

train = pd.read_csv(TRAIN_PATH)
# Loading training data and test data

test = pd.read_csv(TEST_PATH)

train = pd.read_csv(TRAIN_PATH)

item = pd.read_csv(ITEM_PATH)

category = pd.read_csv(CATEGORY_PATH)

shop = pd.read_csv(SHOP_PATH)
df = pd.concat([train, test])
#Loading Sample Files for Submission

sample = pd.read_csv('../input/competitive-data-science-predict-future-sales/sample_submission.csv')

# Confirmation of the format of samples for submission

sample.head(3).style.applymap(lambda x: 'background-color:lightsteelblue')
print('Number of rows in test set: ', test.shape[0])

print('Number of columns in test set: ', test.shape[1])
train.head(5).style.applymap(lambda x: 'background-color:lightsteelblue')
print(train['date_block_num'].max())
print(train['item_cnt_day'].describe())
test.head(5).style.applymap(lambda x: 'background-color:lightsteelblue')
item.head(5).style.applymap(lambda x: 'background-color:lightsteelblue')
category.head(5).style.applymap(lambda x: 'background-color:lightsteelblue')
print(f"{b_}Number of rows in train data: {g_}{train.shape[0]}\n{b_}Number of columns in train data: {g_}{train.shape[1]}")
# Check for missing values in the training features data

train.isnull().sum()
df.info()
# Check age-related statistics in the Training data

train.describe().style.applymap(lambda x: 'background-color:yellow')
# Check age-related statistics in the test data

test.describe().style.applymap(lambda x: 'background-color:lightgreen')
# coding: utf-8

from tqdm import tqdm

import time



# Set the total value 

bar = tqdm(total = 1000)

# Add description

bar.set_description('Progress rate')

for i in range(100):

    # Set the progress

    bar.update(25)

    time.sleep(1)
train.groupby( ['shop_id','item_id'] ).agg( ['mean','std','count'] )
# number of items per cat 

x=item.groupby(['item_category_id']).count()

x=x.sort_values(by='item_id',ascending=False)

x=x.iloc[0:10].reset_index()

x

# plot a graph

plt.figure(figsize=(8,4))

ax= sns.barplot(x.item_category_id, x.item_id, alpha=0.8)

plt.title("Items per Category")

plt.ylabel('Items', fontsize=18)

plt.xlabel('Category', fontsize=18)

plt.show()
# Data Definitions

x = train['date_block_num']

y = train['item_price']



# Title of the chart

plt.title('Price transitions over months', fontsize=20)



# Drawing a Graph

plt.plot(x, y)

plt.show()