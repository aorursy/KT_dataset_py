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

import xgboost as xgb

from sklearn.metrics import log_loss

from sklearn.preprocessing import LabelEncoder

from sklearn import preprocessing

from sklearn.model_selection import KFold

from sklearn.tree import DecisionTreeRegressor
# One-hot encoding

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestRegressor
# Setup the paths to train and test images

DATASET = '../input/osic-pulmonary-fibrosis-progression'

TEST_DIR = '../input/osic-pulmonary-fibrosis-progression/test'

TRAIN_CSV_PATH = '../input/osic-pulmonary-fibrosis-progression/train.csv'



# Glob the directories and get the lists of train and test images

train_fns = glob(DATASET + '*')

test_fns = glob(TEST_DIR + '*')
# Loading training data and test data

train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')

train_dcm = pd.read_csv('../input/osic-image-eda/n_dicom_df.csv')

train_dcm_shp = pd.read_csv('../input/osic-image-eda/shape_df.csv')

train_meta_dcm = pd.read_csv('../input/pulmonary-fibrosis-prep-data/meta_train_data.csv')

test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
# Display of training data

print(train)
#Loading Sample Files for Submission

sample = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')

# Confirmation of the format of samples for submission

sample.head(3)
#Loading Sample Files for Submission

sample = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')
# display the smoking status of the training data without duplicates

print(train['SmokingStatus'].drop_duplicates())
# display the training data without gender duplication

print(train['Sex'].drop_duplicates())
# Display some of the training data

train.head(10)
# Display some of the training data

train_dcm.head(10)
# Display some of the training data

train_dcm_shp.head(10)
# Display some of the training data

train_meta_dcm.head(10)
# Check for missing values in the training data

train.isnull().sum()
# Let's check the max value and the max value for Weeks

print("Minimum number of value for Weeks is: {}".format(train['Weeks'].min()), "\n" +

      "Maximum number of value for Weeks is: {}".format(train['Weeks'].max() ))
# Check the Patient statistics of the training data

train['Patient'].describe()
# Check the FVC statistics of the training data

train['FVC'].describe(percentiles=[0.1,0.2,0.5,0.75,0.9])
# Check age-related statistics in the training data

train['Age'].describe()
# Display of test data

print(test)
# Combine the Patient ID and Week columns

test_patient_weeklist = test['Patient_Week'] = test['Patient'].astype(str)+"_"+test['Weeks'].astype(str)

test2 = test.drop('Patient', axis=1)

test3 = test.drop('Weeks', axis=1)

test4 = test.reindex(columns=['Patient_Week', 'FVC', 'Percent', 'Age', 'Sex', 'SmokingStatus'])

test4.head(7)
# Find the unique number of patient IDs. 

n = train['Patient'].nunique()

print(n)
# First, I'll use Sturgess's formula to find the appropriate number of classes in the histogram 

k = 1 + math.log2(n)
# Display a histogram of the FVC of the training data

sns.distplot(train['FVC'], kde=True, rug=False, bins=int(k)) 

# Graph Title

plt.title('FVC')

# Show Histogram

plt.show() 
# Display a histogram of the age of the training data

sns.distplot(train['Age'], kde=True, rug=False, bins=int(k)) 

# Title of the study data age graph

plt.title('Age')

# Display a histogram of the age of the training data

plt.show() 
# Show the correlation between age and FVC in the training data

sns.scatterplot(data=train, x='Age', y='FVC')
# Produce correlation coefficients between age and FVC of the training data

df = train

df.corr()['Age']['FVC']
# Narrowing down to smokers in the training data to produce a correlation coefficient between age and FVC 

df_smk = train.query('SmokingStatus == "Currently smokes"')



df_smk.corr()['Age']['FVC']
# Scatterplots of age and FVC for training data extracted by smokers

sns.scatterplot(data=df_smk, x='Age', y='FVC')
# Show the correlation between age and FVC in the training data

sns.scatterplot(data=train, x='Percent', y='FVC')
# Compute summary statistics for FVC aggregated by age

df.groupby('Age').describe()['FVC']
# Calculate summary statistics for FVC aggregated by patient ID 

df.groupby('Patient').describe(percentiles=[0.1,0.2,0.5,0.8])['FVC']
df_corr = df.corr()

print(df_corr)
# View the correlation heat map

corr_mat = df.corr(method='pearson')

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
# Draw a pie chart about gender.

plt.pie(train["Sex"].value_counts(),labels=["Male","Female"],autopct="%.1f%%")

plt.title("Ratio of Sex")

plt.show()
# Draw a pie chart about smoking status

plt.pie(train["SmokingStatus"].value_counts(),labels=["Ex-smoker","Never smoked","Currently smokes"],autopct="%.1f%%")

plt.title("SmokingStatus")

plt.show()
print(train[train.FVC < 1651])
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
file_path = "../input/osic-pulmonary-fibrosis-progression/train/ID00023637202179104603099/3.dcm"

dataset = pydicom.dcmread(file_path)

plot_pixel_array(dataset)
file_path = "../input/osic-pulmonary-fibrosis-progression/train/ID00023637202179104603099/5.dcm"

dataset = pydicom.dcmread(file_path)

plot_pixel_array(dataset)
file_path = "../input/osic-pulmonary-fibrosis-progression/train/ID00023637202179104603099/7.dcm"

dataset = pydicom.dcmread(file_path)

plot_pixel_array(dataset)
file_path = "../input/osic-pulmonary-fibrosis-progression/train/ID00023637202179104603099/15.dcm"

dataset = pydicom.dcmread(file_path)

plot_pixel_array(dataset)
print(train[train.FVC > 3874])
file_path = "../input/osic-pulmonary-fibrosis-progression/train/ID00009637202177434476278/11.dcm"

dataset = pydicom.dcmread(file_path)

plot_pixel_array(dataset)
file_path = "../input/osic-pulmonary-fibrosis-progression/train/ID00014637202177757139317/2.dcm"

dataset = pydicom.dcmread(file_path)

plot_pixel_array(dataset)
file_path = "../input/osic-pulmonary-fibrosis-progression/train/ID00032637202181710233084/30.dcm"

dataset = pydicom.dcmread(file_path)

plot_pixel_array(dataset)
file_path = "../input/osic-pulmonary-fibrosis-progression/train/ID00032637202181710233084/35.dcm"

dataset = pydicom.dcmread(file_path)

plot_pixel_array(dataset)
# Create models and train them with training data

train_x = train.drop(['FVC'], axis=1)

train_y = df['FVC']
#Combining training data with data frames with DICOM data and patient IDs as keys

train_x2 = pd.merge(train_x, train_dcm, on='Patient')
# Conversion of category variables to arbitrary values

train_x2['Sex'] = train_x2['Sex'].map({'Male': 0, 'Female': 1})

train_x2['SmokingStatus'] = train_x2['SmokingStatus'].map({'Never smoked': 0, 'Ex-smoker': 1, 'Currently smokes': 2})
# Confirmation of current value

print(train_x2)
# Combine the Patient ID and Week columns

train_x2['Patient_Week'] = train_x2['Patient'].astype(str)+"_"+train_x2['Weeks'].astype(str)

train_x3 = train_x2.drop('Patient', axis=1)

train_x4 = train_x3.drop('Weeks', axis=1)

train_x5 = train_x4.reindex(columns=['Patient_Week', 'Percent', 'Age', 'Sex', 'SmokingStatus', 'n_dicom', 'n_list'])

train_x5.head(7)

# Confirming the converted value

print(train_x5)
# Conversion of category variables to arbitrary values of test data

test2['Sex'] = test['Sex'].map({'Male': 0, 'Female': 1})

test2['SmokingStatus'] = test2['SmokingStatus'].map({'Never smoked': 0, 'Ex-smoker': 1, 'Currently smokes': 2})
test3 = test2.drop('Weeks', axis=1)

test4 = test3.drop('FVC', axis=1)

test5 = test4.reindex(columns=['Patient_Week', 'Percent', 'Age', 'Sex', 'SmokingStatus'])

test5.head(7)

# Confirming the converted value

print(test5)
# I'll just copy test data

test_x = test5.copy()

print(test_x)
osic_features = ['Percent', 'Age', 'Sex', 'SmokingStatus']
X = train_x5[osic_features]
# Define model. Specify a number for random_state to ensure same results each run

osic_model = DecisionTreeRegressor(random_state=1)



# Fit model

osic_model.fit(X, train_y)
print(X.head())

print("The predictions are")

print(osic_model.predict(X.head()))
# Let's visualize the FVC of Training Data

# plt.figure(figsize=(18,6))

# plt.plot(train_x5["FVC"], label = "Train_Data")

# plt.legend()
# Let's visualize the FVC predictions

plt.figure(figsize=(18,6))



Y_train_Graph = pd.DataFrame(X)

plt.plot(Y_train_Graph, label = "Predict")

plt.legend()
#Reading the file

submission = pd.DataFrame(columns = ["Patient_Week", "FVC", "Confidence"])

#Exporting Files

submission.to_csv('submission.csv', index=False)