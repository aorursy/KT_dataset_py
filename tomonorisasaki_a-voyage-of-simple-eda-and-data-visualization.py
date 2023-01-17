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

DATASET = '../input/titanic'

TEST_PATH = '../input/titanic/test.csv'

TRAIN_PATH = '../input/titanic/train.csv'

SAMPLE_PATH = '../input/titanic/gender_submission.csv'



# Loading training data and test data

test = pd.read_csv(TEST_PATH)

train = pd.read_csv(TRAIN_PATH)

sample = pd.read_csv(SAMPLE_PATH)
# Confirmation of the format of samples for submission

sample.head(5).style.applymap(lambda x: 'background-color:lightsteelblue')
HTML('<iframe width="800" height="500" src="https://www.youtube.com/embed/8yZMXCaFshs" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')
print(f"{b_}Number of rows in train data: {r_}{train.shape[0]}\n{b_}Number of columns in train data: {r_}{train.shape[1]}")
train.head(5).style.applymap(lambda x: 'background-color:lightsteelblue')
print(f"{b_}Number of rows in test data: {r_}{test.shape[0]}\n{b_}Number of columns in test data: {r_}{test.shape[1]}")
test.head(5).style.applymap(lambda x: 'background-color:lightsteelblue')
# Check for missing values in the training features data

train.isnull().sum()
# Check statistics in the training data

train.describe().style.applymap(lambda x: 'background-color:yellow')
x = train['Survived'].mean() * 100

survival_rate = math.floor(x)

print('Total number of passengers: ', train['PassengerId'].count())

print('Passenger survival rate: ', survival_rate)
# make sure there are no duplicate passenger IDs.

train.duplicated(['PassengerId']).any()
# Check statistics in the test data

test.describe().style.applymap(lambda x: 'background-color:lightgreen')
# There were missing values for age, so we want to create a traing dataset that excludes the missing values.

train_age = train.dropna(subset=['Age'])
# Find the unique number of patient IDs. 

n = train_age['Age'].nunique()



# First, I'll use Sturgess's formula to find the appropriate number of classes in the histogram 

k = 1 + math.log2(n)



# Display a histogram of the FVC of the training data

sns.distplot(train_age['Age'], kde=True, rug=False, bins=int(k)) 

# Graph Title

plt.title('Age distribution of Titanic passengers')

# Show Histogram

plt.show() 
# Display with comparison by age

g = sns.FacetGrid(train_age,hue="Sex",height=5)

g.map(sns.distplot, "Age", kde=False)

g.add_legend()
# Draw a pie chart about gender.

plt.pie(train["Sex"].value_counts(),labels=["Male","Female"],autopct="%.1f%%")

plt.title("Ratio of Sex of Titanic passengers")

plt.show()
# Draw a pie chart about Passenger class

plt.pie(train["Pclass"].value_counts(),labels=["1","2","3"],autopct="%.1f%%")

plt.title("Pclass")

plt.show()
# Display a bar graph of passenger classes by gender

sns.countplot(x="Pclass", hue="Sex",data=train_age)
# Extract data on children and the elderly

train_children = train_age.query('Age < 20')

train_seniors = train_age.query('Age > 60')



# Combinine the extracted data for children and elderly

train_c_and_s = pd.concat([train_children, train_seniors])
# Calculate passenger numbers and survival rates from data on children and the elderly

r = train_c_and_s['Survived'].mean() * 100

survival_rate = math.floor(r)

print('Total number of passengers: ', train_c_and_s['PassengerId'].count())

print('Passenger survival rate: ', survival_rate)
# Extract data for first and third class passengers

train_1st = train.query('Pclass == 1')

train_3rd = train.query('Pclass == 3')



# Calculate passenger numbers and survival rates from data on Pclass

r1 = train_1st['Survived'].mean() * 100

r3 = train_3rd['Survived'].mean() * 100

survival_rate_1st = math.floor(r1)

survival_rate_3rd = math.floor(r3)

print('Total number of 1st class passengers: ', train_1st['PassengerId'].count())

print('1st class Passenger survival rate: ', survival_rate_1st)

print('Total number of 3rd class passengers: ', train_3rd['PassengerId'].count())

print('3rd class passenger survival rate: ', survival_rate_3rd)
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