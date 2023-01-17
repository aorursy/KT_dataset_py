import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
!conda install -c conda-forge gdcm -y
import pydicom

import cv2

import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf

from tensorflow.keras.layers import Dense , Conv3D , MaxPool3D , BatchNormalization , Dropout , Concatenate ,Input , Flatten

from tensorflow.keras.models import Model , Sequential

import PIL

import gdcm
train_dir = os.path.join('../input/osic-pulmonary-fibrosis-progression/train/')

print('train_dir = ' , train_dir)

test_dir = os.path.join('../input/osic-pulmonary-fibrosis-progression/test/')

print('test_dir = ' , test_dir)
#training data

train_labels = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')

print(train_labels.shape)

train_labels.head()
test_labels = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

print(test_labels.shape)

test_labels.head()
train_patients = train_labels['Patient'].values

train_patients.shape
count = 0

for p in train_patients[27:28]:

        path = train_dir + p

        slices = [pydicom.dcmread(path+'/'+s) for s in os.listdir(path)]

        slices.sort(key = lambda x: int(x.InstanceNumber))

        a = slices[0].pixel_array.shape

        #print(len(slices) , slices[0].pixel_array.shape)

        #print(len(slices))

        print(slices[0])

#     except RuntimeError as e:

#         count += 1



IMG_SIZE = 150 



for p in train_patients[:1]:

    path = train_dir + p

    slices = [pydicom.read_file(path+'/'+s) for s in os.listdir(path)]

    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))

    fig = plt.figure(figsize=(8 , 8))

    for i , scan in enumerate(slices[:12]):

        p = fig.add_subplot(3, 4 , i+1)

        img = cv2.resize(np.array(scan.pixel_array) , (IMG_SIZE , IMG_SIZE))

        p.imshow(img , cmap = 'gray')

    plt.show()

        
NUM_SLICES = 30



def chunks(lst, n):

    """Yield successive n-sized chunks from lst."""

    for i in range(0, len(lst), n):

        yield lst[i:i + n]

        

def mean(l):

    return sum(l)/len(l)





def resize_slices(slices):

    slices = [cv2.resize(np.array(slice.pixel_array) , (IMG_SIZE , IMG_SIZE)) for slice in slices]

    if len(slices) == NUM_SLICES:

        return slices

    else:

        chunk_size = int(np.ceil(len(slices) / NUM_SLICES))

        new_slices = []

        for chunk in chunks(slices , chunk_size):

            chunk = list(map(mean , zip(*chunk)))

            new_slices.append(chunk)

        if len(new_slices) < NUM_SLICES:

            for i in range(NUM_SLICES - len(new_slices)):

                new_slices.append(new_slices[-1])

        elif len(new_slices) > NUM_SLICES:

            extra = new_slices[NUM_SLICES-1:]

            last = list(map(mean , zip(*extra)))

            del new_slices[NUM_SLICES:]

            new_slices[-1] = last

        return new_slices

        
for p in train_patients[22:28]:

    path = train_dir + p

    slices = [pydicom.read_file(path+'/'+s) for s in os.listdir(path)]

    slices.sort(key = lambda x: int(x.InstanceNumber))

    new_slices = resize_slices(slices)

    print(np.shape(new_slices))

    
train_labels.columns
num_smoke_classes = train_labels['SmokingStatus'].unique()

print(len(num_smoke_classes))

print(num_smoke_classes)
train_labels.isnull().sum()
categorical_attribute = train_labels['SmokingStatus'].tolist()

for i in range(len(categorical_attribute)):

    if categorical_attribute[i] == 'Ex-smoker':

        categorical_attribute[i] = [0,0,1]

    elif categorical_attribute[i] == 'Never smoked':

        categorical_attribute[i] = [0,1,0]

    elif categorical_attribute[i] == 'Currently smokes':

        categorical_attribute[i] = [1,0,0]

categorical_attribute = np.array(categorical_attribute)

print(categorical_attribute[:10])

print(categorical_attribute.shape)
numerical_features = ['Weeks' , 'Age' , 'Sex' , 'SmokingStatus']

numerical_data = train_labels[['Weeks','Age' ,]]

numerical_data = np.array(numerical_data)

numerical_data.shape
numerical_data[:4]
numerical_data = np.concatenate((numerical_data , categorical_attribute) , axis = 1)

print(numerical_data.shape)

numeraical_data = np.array(numerical_data)

numerical_data[:10]
print(train_labels['Sex'].unique())
gender_oh = train_labels['Sex'].tolist()

for i in range(len(gender_oh)):

    if gender_oh[i] == 'Male':

        gender_oh[i] = [0,1]

    elif gender_oh[i] == 'Female':

        gender_oh[i] = [1,0]

gender_oh = np.array(gender_oh)

print(gender_oh[:10])

print(gender_oh.shape)
numerical_data = np.concatenate((numerical_data , gender_oh) , axis = 1)

print(numerical_data.shape)

numeraical_data = np.array(numerical_data)

numerical_data[:10]
numerical_data[:20]
np.save('Train_X_numerical.npy' , numerical_data)
pydicom.config.image_handlers = ['pillow_handler']
ct_data = []

for p in train_patients:

    path = train_dir + p

    slices = [pydicom.read_file(path+'/'+s) for s in os.listdir(path)]

    slices.sort(key = lambda x: int(x.InstanceNumber))

    try:

        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])

    except:

        slice_thickness = slices[0].SliceThickness

    for s in slices:

        s.SliceThickness = slice_thickness

    new_slices = resize_slices(slices)

    new_slices = np.array(new_slices).astype(np.int16)

    ct_data.append(new_slices)



ct_data = np.array(ct_data)

print(ct_data.shape)
ct_data = np.array(ct_data)

np.save('Train_ct_processed.npy' , ct_data)