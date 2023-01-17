from IPython.display import HTML
HTML('<iframe width="800" height="500" src="https://www.youtube.com/embed/4C6BB56fG1M" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')
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



from PIL import Image





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

DATASET = '../input/rsna-str-pulmonary-embolism-detection'

TEST_DIR = '../input/rsna-str-pulmonary-embolism-detection/test/'

TRAIN_DIR = '../input/rsna-str-pulmonary-embolism-detection/'

SCAN_DIR = '../input/pulmonary-embolism-ct-data/'
# Display some of the training data

train = pd.read_csv(TRAIN_DIR + "train.csv")

train.head(10).style.applymap(lambda x: 'background-color:lightsteelblue')
print(f"{b_}Number of rows in sample data: {r_}{train.shape[0]}\n{b_}Number of columns in sample data: {r_}{train.shape[1]}")
# Display some of the training data

train.info()
# Display some of the scan data

scan = pd.read_csv(SCAN_DIR + "Pulmonary_Embolism_CT_scans_data.csv")

scan.head(5).style.applymap(lambda x: 'background-color:lightsteelblue')
print(f"{b_}Number of rows in sample data: {r_}{scan.shape[0]}\n{b_}Number of columns in sample data: {r_}{scan.shape[1]}")
sample = pd.read_csv(TRAIN_DIR + "sample_submission.csv")

# Confirmation of the format of samples for submission

sample.head(3).style.applymap(lambda x: 'background-color:lightsteelblue')
print('The number of SOPInstanceUID is ' + str(len(train['SOPInstanceUID'].unique())))

print('The number of StudyInstanceUID is ' + str(len(train['StudyInstanceUID'].unique())))

print('The number of SeriesInstanceUID is ' + str(len(train['SeriesInstanceUID'].unique())))
# display IDs of the training data without duplicates

print(train['SOPInstanceUID'].drop_duplicates())

print(train['StudyInstanceUID'].drop_duplicates())

print(train['SeriesInstanceUID'].drop_duplicates())
# Display of test data

test = pd.read_csv(TRAIN_DIR + "test.csv")

test.head(10).style.applymap(lambda x: 'background-color:lightsteelblue')
# Display of test data

test.info(10)
# Check for missing values in the training data

train.isnull().sum()
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
# Let's check the max value and the max value for pe_present_on_image

print("Minimum number of value for pe_present_on_image is: {}".format(train['pe_present_on_image'].min()), "\n" +

      "Maximum number of value for pe_present_on_image is: {}".format(train['pe_present_on_image'].max() ))

# Let's check the max value and the max value for negative_exam_for_pe

print("Minimum number of value for negative_exam_for_pe is: {}".format(train['negative_exam_for_pe'].min()), "\n" +

      "Maximum number of value for negative_exam_for_pe is: {}".format(train['negative_exam_for_pe'].max() ))

# Let's check the max value and the max value for qa_motion

print("Minimum number of value for qa_motion is: {}".format(train['qa_motion'].min()), "\n" +

      "Maximum number of value for qa_motion is: {}".format(train['qa_motion'].max() ))

# Let's check the max value and the max value for qa_contrast

print("Minimum number of value for qa_contrast is: {}".format(train['qa_contrast'].min()), "\n" +

      "Maximum number of value for qa_contrast is: {}".format(train['qa_contrast'].max() ))

# Let's check the max value and the max value for flow_artifact

print("Minimum number of value for flow_artifact is: {}".format(train['flow_artifact'].min()), "\n" +

      "Maximum number of value for flow_artifact is: {}".format(train['flow_artifact'].max() ))

# Let's check the max value and the max value for rv_lv_ratio_gte_1

print("Minimum number of value for rv_lv_ratio_gte_1 is: {}".format(train['rv_lv_ratio_gte_1'].min()), "\n" +

      "Maximum number of value for rv_lv_ratio_gte_1 is: {}".format(train['rv_lv_ratio_gte_1'].max() ))

# Let's check the max value and the max value for rv_lv_ratio_lt_1

print("Minimum number of value for rv_lv_ratio_lt_1 is: {}".format(train['rv_lv_ratio_lt_1'].min()), "\n" +

      "Maximum number of value for rv_lv_ratio_lt_1 is: {}".format(train['rv_lv_ratio_lt_1'].max() ))

# Let's check the max value and the max value for leftsided_pe

print("Minimum number of value for leftsided_pe is: {}".format(train['leftsided_pe'].min()), "\n" +

      "Maximum number of value for leftsided_pe is: {}".format(train['leftsided_pe'].max() ))

# Let's check the max value and the max value for true_filling_defect_not_pe

print("Minimum number of value for true_filling_defect_not_pe is: {}".format(train['true_filling_defect_not_pe'].min()), "\n" +

      "Maximum number of value for true_filling_defect_not_pe is: {}".format(train['true_filling_defect_not_pe'].max() ))

# Let's check the max value and the max value for rightsided_pe

print("Minimum number of value for rightsided_pe is: {}".format(train['rightsided_pe'].min()), "\n" +

      "Maximum number of value for rightsided_pe is: {}".format(train['rightsided_pe'].max() ))

# Let's check the max value and the max value for central_pe

print("Minimum number of value for central_pe is: {}".format(train['central_pe'].min()), "\n" +

      "Maximum number of value for central_pe is: {}".format(train['central_pe'].max() ))

# Let's check the max value and the max value for indeterminate

print("Minimum number of value for rightsided_pe is: {}".format(train['indeterminate'].min()), "\n" +

      "Maximum number of value for rightsided_pe is: {}".format(train['indeterminate'].max() ))
# Draw a pie chart about pe_present_on_image.

plt.pie(train["pe_present_on_image"].value_counts(),labels=["0","1"],autopct="%.1f%%")

plt.title("Ratio of pe_present_on_image")

plt.show()
# Draw a pie chart about qa_motion.

plt.pie(train["qa_motion"].value_counts(),labels=["0","1"],autopct="%.1f%%")

plt.title("Ratio of qa_motion")

plt.show()
# Draw a pie chart about negative_exam_for_pe.

plt.pie(train["negative_exam_for_pe"].value_counts(),labels=["0","1"],autopct="%.1f%%")

plt.title("Ratio of negative_exam_for_pe")

plt.show()
# Draw a pie chart about qa_contrast.

plt.pie(train["qa_contrast"].value_counts(),labels=["0","1"],autopct="%.1f%%")

plt.title("Ratio of qa_contrast")

plt.show()
# Draw a pie chart about flow_artifact.

plt.pie(train["flow_artifact"].value_counts(),labels=["0","1"],autopct="%.1f%%")

plt.title("Ratio of flow_artifact")

plt.show()
# Draw a pie chart about rv_lv_ratio_gte_1.

plt.pie(train["rv_lv_ratio_gte_1"].value_counts(),labels=["0","1"],autopct="%.1f%%")

plt.title("Ratio of rv_lv_ratio_gte_1")

plt.show()
# Draw a pie chart about rv_lv_ratio_lt_1.

plt.pie(train["rv_lv_ratio_lt_1"].value_counts(),labels=["0","1"],autopct="%.1f%%")

plt.title("Ratio of rv_lv_ratio_lt_1")

plt.show()
# Draw a pie chart about Ratio of leftsided_pe.

plt.pie(train["leftsided_pe"].value_counts(),labels=["0","1"],autopct="%.1f%%")

plt.title("Ratio of leftsided_pe")

plt.show()
# Draw a pie chart about Ratio of true_filling_defect_not_pe.

plt.pie(train["true_filling_defect_not_pe"].value_counts(),labels=["0","1"],autopct="%.1f%%")

plt.title("Ratio of true_filling_defect_not_pe")

plt.show()
# Draw a pie chart about Ratio of rightsided_pe.

plt.pie(train["rightsided_pe"].value_counts(),labels=["0","1"],autopct="%.1f%%")

plt.title("Ratio of rightsided_pe")

plt.show()
# Draw a pie chart about Ratio of central_pe.

plt.pie(train["central_pe"].value_counts(),labels=["0","1"],autopct="%.1f%%")

plt.title("Ratio of central_pe")

plt.show()
# Draw a pie chart about Ratio of indeterminate.

plt.pie(train["indeterminate"].value_counts(),labels=["0","1"],autopct="%.1f%%")

plt.title("Ratio of indeterminate")

plt.show()
# Find the unique number of pixelspacing_area. 

n = scan['pixelspacing_area'].nunique()

# First, I'll use Sturgess's formula to find the appropriate number of classes in the histogram 

k = 1 + math.log2(n)

# Display a histogram of the pixelspacing_area of the training data

sns.distplot(scan['pixelspacing_area'], kde=True, rug=False, bins=int(k)) 

# Graph Title

plt.title('pixelspacing_area')

# Show Histogram

plt.show() 
# Find the unique number of pixelspacing_c. 

n = scan['pixelspacing_c'].nunique()

# First, I'll use Sturgess's formula to find the appropriate number of classes in the histogram 

k = 1 + math.log2(n)

# Display a histogram of the pixelspacing_c of the training data

sns.distplot(scan['pixelspacing_c'], kde=True, rug=False, bins=int(k)) 

# Graph Title

plt.title('pixelspacing_c')

# Show Histogram

plt.show() 
# Find the unique number of pixelspacing_r. 

n = scan['pixelspacing_r'].nunique()

# First, I'll use Sturgess's formula to find the appropriate number of classes in the histogram 

k = 1 + math.log2(n)

# Display a histogram of the pixelspacing_c of the training data

sns.distplot(scan['pixelspacing_r'], kde=True, rug=False, bins=int(k)) 

# Graph Title

plt.title('pixelspacing_r')

# Show Histogram

plt.show() 
# View the correlation heat map

corr_mat = train.corr(method='pearson')

sns.heatmap(corr_mat,

            vmin=-1.0,

            vmax=1.0,

            center=0,

            annot=True, # True:Displays values in a grid

            fmt='.1f',

            xticklabels=corr_mat.columns.values,

            yticklabels=corr_mat.columns.values

           )

plt.show()
def extract_num(s, p, ret=0):

    search = p.search(s)

    if search:

        return int(search.groups()[0])

    else:

        return ret
import pydicom



def plot_pixel_array(dataset, figsize=(5,5)):

    plt.figure(figsize=figsize)

    plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)

    plt.show()
file_path = "../input/rsna-str-pulmonary-embolism-detection/train/0003b3d648eb/d2b2960c2bbf/00ac73cfc372.dcm"

dataset = pydicom.dcmread(file_path)

plot_pixel_array(dataset)