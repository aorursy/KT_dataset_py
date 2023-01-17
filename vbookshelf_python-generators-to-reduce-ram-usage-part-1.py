from numpy.random import seed
seed(101)
from tensorflow import set_random_seed
set_random_seed(101)

import pandas as pd
import numpy as np
import pydicom
import pylab
import os
import pickle

import matplotlib.pyplot as plt
%matplotlib inline

# Don't Show Warning Messages
import warnings
warnings.filterwarnings('ignore')

df_train = pd.read_csv('../input/stage_1_train_labels.csv')
df_test = pd.read_csv('../input/stage_1_sample_submission.csv')
df_info = pd.read_csv('../input/stage_1_detailed_class_info.csv')

print(df_train.shape)
print(df_test.shape)
print(df_info.shape)

# check df_info for missing values
df_info.isnull().sum()
# New Col: num_bounding_boxes
df_train['num_bounding_boxes'] = 1
# create a dataframe with unique patientId's
df_group = \
df_train.drop(['x','y','width','height','Target'], axis=1).groupby('patientId').sum()

# reset the index
df_group.reset_index(inplace=True)

df_group.head()
# check
print(df_train['patientId'].nunique())
print(len(df_group))
# create new columns
df_group['PatientAge'] = 0
df_group['PatientSex'] = 0
df_group['ViewPosition'] = 0
# extract the meta data and store in df_group

for i in range(0,len(df_group)):
    patientId = df_group.loc[i,'patientId']
    
    path = \
'../input/stage_1_train_images/%s.dcm' % patientId
    
    # get the meta data
    dcm_data = pydicom.read_file(path)
    
    df_group.loc[i,'PatientAge'] = dcm_data.PatientAge
    df_group.loc[i,'PatientSex'] = dcm_data.PatientSex
    df_group.loc[i,'ViewPosition'] = dcm_data.ViewPosition
    
# create new columns
df_test['PatientAge'] = 0
df_test['PatientSex'] = 0
df_test['ViewPosition'] = 0

for i in range(0,len(df_test)):
    patientId = df_test.loc[i,'patientId']
    
    path = \
'../input/stage_1_test_images/%s.dcm' % patientId
    
    # get the meta data
    dcm_data = pydicom.read_file(path)
    
    df_test.loc[i,'PatientAge'] = dcm_data.PatientAge
    df_test.loc[i,'PatientSex'] = dcm_data.PatientSex
    df_test.loc[i,'ViewPosition'] = dcm_data.ViewPosition
    
# change the datatype to int16. now it is string
df_group['PatientAge'] = df_group['PatientAge'].astype(np.int16)
df_test['PatientAge'] = df_test['PatientAge'].astype(np.int16)
df = df_info

df.drop_duplicates(inplace=True)

# reset the index or NaN's will be produced when trying to add the class col to df_group
df.reset_index(inplace=True)

df_group['class'] = df['class']
# check for age errors in the train set
df_group[df_group['PatientAge'] > 100]

# 5 age errors found
# check for age errors in the test set
df_test[df_test['PatientAge'] > 100]

# no age errors found
# view the x-rays
# load a patient's file
patientId = df_train.loc[24537,'patientId']
path = \
'../input/stage_1_train_images/%s.dcm' % patientId

dcm_data = pydicom.read_file(path)

# convert the image to a numpy array
im = dcm_data.pixel_array

# view an image
pylab.imshow(im, cmap=pylab.cm.gist_gray)
pylab.axis('off')
# remove the 1 at the start of the age
# assumes 1 was added by mistake to these ages
age_errors = ['3b8b8777-a1f6-4384-872a-28b95f59bf0d', '73aeea88-fc48-4030-8564-0a9d7fdecac4',
             'a4e8e96d-93a6-4251-b617-91382e610fab', 'ec3697bd-184e-44ba-9688-ff8d5fbf9bbc',
             'f632328d-5819-4b29-b54f-adf4934bbee6']

df_group.loc[3175, 'PatientAge'] = 48
df_group.loc[9708, 'PatientAge'] = 51
df_group.loc[15273, 'PatientAge'] = 53
df_group.loc[23374, 'PatientAge'] = 50
df_group.loc[24537, 'PatientAge'] = 55
# check the changes
df_group.loc[23374,:]
df_group['noopacity_but_not_normal'] = df_group['class']

def noopacity_but_not_normal(x):
    if x == 'No Lung Opacity / Not Normal':
        return 1
    else:
        return 0

df_group['noopacity_but_not_normal'] = \
df_group['noopacity_but_not_normal'].apply(noopacity_but_not_normal)
# New Col: target
df_group['target'] = \
df_group['class'].map({'Lung Opacity':1, 'Normal':0, 'No Lung Opacity / Not Normal':0})
for i in range(0,len(df_group)):
    if df_group.loc[i,'target'] == 0:
        df_group.loc[i,'num_bounding_boxes'] = 0
# Source: https://www.kaggle.com/peterchang77/exploratory-data-analysis

def parse_data(df):
    """
    Method to read a CSV file (Pandas dataframe) and parse the 
    data into the following nested dictionary:

      parsed = {
        
        'patientId-00': {
            'dicom': path/to/dicom/file,
            'label': either 0 or 1 for normal or pnuemonia, 
            'boxes': list of box(es)
        },
        'patientId-01': {
            'dicom': path/to/dicom/file,
            'label': either 0 or 1 for normal or pnuemonia, 
            'boxes': list of box(es)
        }, ...

      }

    """
    # --- Define lambda to extract coords in list [y, x, height, width]
    extract_box = lambda row: [row['y'], row['x'], row['height'], row['width']]

    parsed = {}
    for n, row in df.iterrows():
        # --- Initialize patient entry into parsed 
        pid = row['patientId']
        if pid not in parsed:
            parsed[pid] = {
                'dicom': '../input/stage_1_train_images/%s.dcm' % pid,
                'label': row['Target'],
                'boxes': []}

        # --- Add box if opacity is present
        if parsed[pid]['label'] == 1:
            parsed[pid]['boxes'].append(extract_box(row))

    return parsed


# run the function

df_boxes = pd.read_csv('../input/stage_1_train_labels.csv')

parsed = parse_data(df_boxes)
# check the bounding boxes for one patientId
box = parsed['00436515-870c-4b36-a041-de91049b9ab4']['boxes']

box
# extract the boxes for each patient

df_group['bounding_boxes'] = df_group['patientId']

def bounding_boxes(x):
    # get the dictionary value
    box = parsed[x]['boxes']
    
    return box

df_group['bounding_boxes'] = df_group['bounding_boxes'].apply(bounding_boxes)
# drop the PredictionString col from df_test
df_test = df_test.drop('PredictionString', axis=1)
df_group.isnull().sum()        
df_test.isnull().sum()
# note: we are saving df_group with the name dftrain for easy reference later
pickle.dump(df_group,open('dftrain.pickle','wb'))
pickle.dump(df_test,open('dftest.pickle','wb'))
# check if the pickled files exist
!ls
