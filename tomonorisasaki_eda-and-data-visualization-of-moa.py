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

from sklearn.decomposition import PCA

from IPython.display import HTML

from PIL import Image



# import data visualization

import matplotlib.pyplot as plt

import matplotlib.patches as patches

import seaborn as sns

import plotly.express as px



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

DATASET = '../input/lish-moa'

TEST_DIR_CSV_PATH = '../input/lish-moa/test_features.csv'

TEST_FTR_PATH = '../input/lish-moa/test_features.csv'

TRAIN_FTR_PATH = '../input/lish-moa/train_features.csv'

TRAIN_CSV_NON_PATH = '../input/lish-moa/train_targets_nonscored.csv'

TRAIN_CSV_SCR_PATH = '../input/lish-moa/train_targets_scored.csv'
# Loading training data and test data

test_ftr = pd.read_csv(TEST_FTR_PATH)

train_ftr = pd.read_csv(TRAIN_FTR_PATH)

train_csv_non = pd.read_csv(TRAIN_CSV_NON_PATH)

train_csv_scr = pd.read_csv(TRAIN_CSV_SCR_PATH)
df = pd.concat([train_ftr, test_ftr])
#Loading Sample Files for Submission

sample = pd.read_csv('../input/lish-moa/sample_submission.csv')

# Confirmation of the format of samples for submission

sample.head(3).style.applymap(lambda x: 'background-color:lightsteelblue')
HTML('<iframe width="800" height="500" src="https://www.youtube.com/embed/UMxsZdVrA7A" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')

print('Number of rows in test set: ', test_ftr.shape[0])

print('Number of columns in test set: ', test_ftr.shape[1] - 1)
test_ftr.head(5).style.applymap(lambda x: 'background-color:lightsteelblue')
print('Number of rows in training set: ', train_ftr.shape[0])

print('Number of columns in training set: ', train_ftr.shape[1] - 1)
train_ftr.head(5).style.applymap(lambda x: 'background-color:lightsteelblue')
print(f"{b_}Number of rows in train data: {r_}{train_ftr.shape[0]}\n{b_}Number of columns in train data: {r_}{train_ftr.shape[1]}")
train_csv_non.head(5).style.applymap(lambda x: 'background-color:lightsteelblue')
print(f"{b_}Number of rows in train data: {r_}{train_csv_non.shape[0]}\n{b_}Number of columns in train data: {r_}{train_csv_non.shape[1]}")
train_csv_scr.head(5).style.applymap(lambda x: 'background-color:lightsteelblue')
print(f"{b_}Number of rows in train data: {r_}{train_csv_scr.shape[0]}\n{b_}Number of columns in train data: {r_}{train_csv_scr.shape[1]}")
# Check for missing values in the training features data

train_ftr.isnull().sum()
# Check for missing values in the training targets nonscored data

train_csv_non.isnull().sum()
# Check for missing values in the training targets scored data

train_csv_scr.isnull().sum()
df.info()
# Number of Gene expression columns

train_ftr.columns.str.startswith('g-').sum()
# Number of Cell viability columns

train_ftr.columns.str.startswith('c-').sum()
# Check age-related statistics in the Training data

train_ftr.describe().style.applymap(lambda x: 'background-color:yellow')
# Check age-related statistics in the test data

test_ftr.describe().style.applymap(lambda x: 'background-color:lightgreen')
train_ftr.groupby( ['cp_type','cp_time','cp_dose'] ).agg( ['mean','std','count'] )
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
plt.figure(figsize=(16, 16))

cols = [

    'c-1', 'c-2', 'c-3', 'c-4',

    'c-5', 'c-6', 'c-7', 'c-8',

    'c-92', 'c-93', 'c-94', 'c-95', 

    'c-96', 'c-97', 'c-98', 'c-99']

for i, col in enumerate(cols):

    plt.subplot(4, 4, i + 1)

    plt.hist(train_ftr.loc[:, col], bins=100, alpha=1,color='#00FFFF');

    plt.title(col)
plt.figure(figsize=(16, 16))

cols = [

    'g-1', 'g-2', 'g-3', 'g-4',

    'g-5', 'g-6', 'g-7', 'g-8',

    'g-92', 'g-93', 'g-94', 'g-95', 

    'g-96', 'g-97', 'g-98', 'g-99']

for i, col in enumerate(cols):

    plt.subplot(4, 4, i + 1)

    plt.hist(train_ftr.loc[:, col], bins=100, alpha=1,color='#800080');

    plt.title(col)
# Draw a pie chart about CPtypes of Training data.

plt.pie(train_ftr["cp_type"].value_counts(),labels=["trt_cp","ctl_vehicle"],autopct="%.1f%%")

plt.title("Ratio of CPtypes of Training data")

plt.show()
plt.figure(figsize=(15,5))

sns.distplot(train_ftr['cp_time'], color='blue', bins=10)

plt.title("Train: Treatment duration ", fontsize=15, weight='bold')

plt.show()
# Draw a pie chart about CPtypes of Training data.

plt.pie(train_ftr["cp_dose"].value_counts(),labels=["D1","D2"],autopct="%.1f%%")

plt.title("Ratio of CPdose")

plt.show()
# Set the total value 

bar = tqdm(total = 1000)

# Add description

bar.set_description('Progress rate')

for i in range(100):

    # Set the progress

    bar.update(25)

    time.sleep(1)
plt.figure(figsize=(5,12))

plt.subplot(3,1,1)

splot = sns.countplot(train_ftr["cp_type"],color='#33FFCC')

for p in splot.patches:

    splot.annotate(format(p.get_height(), '.1f'), 

                   (p.get_x() + p.get_width() / 2., p.get_height()), 

                   ha = 'center', va = 'center', 

                   xytext = (0, 9), 

                   textcoords = 'offset points')

plt.title('cp_type')

plt.subplot(3,1,2)

sns.countplot(train_ftr['cp_time'],hue=train_ftr['cp_type'],color='#00FF00')

plt.title('cp_time and cp_type')

plt.subplot(3,1,3)

sns.countplot(train_ftr['cp_dose'],hue=train_ftr['cp_type'],color='#00FFFF')

plt.title('cp_dose and cp_type')

plt.tight_layout()
# Calculate Pearson's r using pandas

res=train_ftr.corr() 



# Show correlation matrix

print(res)
# View the heat map of the correlation matrix

sns.heatmap(res,square=True)