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



# import data visualization

import matplotlib.pyplot as plt

import matplotlib.patches as patches

import seaborn as sns



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

from xgboost import XGBClassifier

from sklearn import preprocessing
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



# import data visualization

import matplotlib.pyplot as plt

import matplotlib.patches as patches

import seaborn as sns



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

from xgboost import XGBClassifier

from sklearn import preprocessing
# Setup the paths to train and test images

DATASET = '../input/delg-saved-models/'

TEST_DIR = '../input/landmark-recognition-2020/test/'

TRAIN_DIR = '../input/landmark-recognition-2020/'
#Loading train Files for Submission

train = pd.read_csv(TRAIN_DIR + "train.csv")

#Loading Sample Files for Submission

sample = pd.read_csv(TRAIN_DIR + "sample_submission.csv")
# Display some of the training data

train.head(10).style.applymap(lambda x: 'background-color:lightsteelblue')
# Confirmation of the format of samples for submission

sample.head(10).style.applymap(lambda x: 'background-color:lightsteelblue')
#Check for missing values in the training data

train.isnull().sum()
# Find the unique number of landmark IDs. 

n = train['landmark_id'].nunique()

print('The unique number of landmark IDs is ' + str(n))
# First, I'll use Sturgess's formula to find the appropriate number of classes in the histogram 

k = 1 + math.log2(n)
# Display a histogram of the FVC of the training data

sns.distplot(train['landmark_id'], kde=True, rug=False, bins=int(k), color='c') 

# Graph Title

plt.title('Distribuition of landmark_ids')

# label

plt.xlabel("landmark_ids")

plt.ylabel("Frequency")

# Show Histogram

plt.show() 
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
print(train['landmark_id'].value_counts())
s_bool = train['landmark_id'] == 138982

m = s_bool.sum()

print('The number of landmark ID 138982' + str(m))
the_most_cmn_pics = train[train["landmark_id"]==138982]
print(the_most_cmn_pics)