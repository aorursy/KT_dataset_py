# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
pwd

ls /kaggle/working
class_path='/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_detailed_class_info.csv'
labels_path='/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv'
Image_train_path='/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images/'
Image_test_path='/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_test_images/'
DF_class=pd.read_csv(class_path)
DF_class.head(10)
print(DF_class.shape[0])

print(DF_class['patientId'].value_counts().shape[0],'patient cases')
DF_class.groupby('class').size()
DF_class.head()
# Analysis of stage 2 patients labels information
DF_Label=pd.read_csv(labels_path)
print(DF_Label.shape[0])

print(DF_Label['patientId'].value_counts().shape[0])
DF_Label.head()
#Lets merge two data's
DataFrame=pd.merge(DF_class,DF_Label,on='patientId')
print(DataFrame.shape[0])
# Now lets drop the duplicate cases
DataFrame_Comb=pd.concat([DF_Label,DF_class.drop('patientId',1)],1)
print(DataFrame_Comb.shape[0])

DataFrame_Comb.sample(10)

DataFrame_Comb.shape
# Classes and Targets based on Patient count
DataFrame_Comb.groupby(['class','Target']).size().reset_index(name='patient_numbers')

import pydicom
import pylab
import matplotlib.pyplot as plt
import seaborn as sn
# Now lets read the image
# First lets check the image of a person with no lung opacity but not normal
DF_class.iloc[0]
image_path='/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images/0004cfab-14fd-4e49-80ba-63a80b6bddd6.dcm'
Image_1=pydicom.read_file(image_path)
Image_1
Image_1.pixel_array.shape
# lets check the view of the lung
plt.figure(figsize=(12,10))
plt.subplot(121)
plt.title('color scale image')
plt.imshow(Image_1.pixel_array)
plt.subplot(122)
plt.title('gray scale image')
plt.imshow(Image_1.pixel_array,cmap=plt.cm.gist_gray)
# Patient who is normal and image
DF_class.iloc[3]
image_path_1='/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images/0004cfab-14fd-4e49-80ba-63a80b6bddd6.dcm'
dcm_1_data=pydicom.read_file(image_path_1)
dcm_1_data
dcm_1_data.pixel_array.shape
# lets check the view of the lung
plt.figure(figsize=(12,10))
plt.subplot(121)
plt.title('color scale image')
plt.imshow(dcm_1_data.pixel_array)
plt.subplot(122)
plt.title('gray scale image')
plt.imshow(dcm_1_data.pixel_array,cmap=plt.cm.gist_gray)
# Now finally lets check the patient who is having pneumonia
DF_class.iloc[4]
image_path_2='/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images/00436515-870c-4b36-a041-de91049b9ab4.dcm'
dcm_2_data=pydicom.read_file(image_path_2)
dcm_2_data
dcm_2_data.pixel_array.shape
plt.figure(figsize=(12,10))
plt.subplot(121)
plt.title('color scale image')
plt.imshow(dcm_2_data.pixel_array)
plt.subplot(122)
plt.title('gray scale image')
plt.imshow(dcm_2_data.pixel_array,cmap=plt.cm.gist_gray)
# Lets check how many images are there
image_data=os.listdir(Image_train_path)
len(image_data)

#getting useful information from images
patient_data=[]
for i in DF_Label['patientId']:
    patient_data_path='/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images/%s.dcm' % i
    patient_image_data=pydicom.read_file(patient_data_path)
    patient_data.append([i,
                         patient_image_data.PatientAge,
                         patient_image_data.PatientSex,
                         patient_image_data.ViewPosition,
                         patient_image_data.Rows,
                         patient_image_data.Columns])

patient_data[:5]
DF_patient_data=pd.DataFrame(data=patient_data,columns=['patientId','patientAge','patientSex','patient_View_position',
                                                        'pixel_rows','pixel_columns'])
DF_patient_data.head()
DF_patient_data['patientAge']=DF_patient_data['patientAge'].apply(int)
DF_patient_data.shape
#Now lets combine all the dataset
DF_Full=pd.concat([DF_patient_data,DataFrame_Comb],axis=1)
DF_Full.head()
DF_Full.shape
# Lets drop the duplicate columns
DF_Full=DF_Full.loc[:,~DF_Full.columns.duplicated()]
DF_Full.shape
DF_Full.describe()
DF_Full.isnull().sum()
DF_Full['class'].value_counts()
DF_Full['Target'].value_counts()
DF_Full['patientSex'].value_counts()
from skimage.transform import resize
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
# Building the model
DF_Full.shape
resized_shape=(64,64)
patientId=DF_Full['patientId'][1]
print(DF_Full['patientId'][1])
# Creating pixel columns
pixel_labels=[]
for i in range(resized_shape[0]*resized_shape[1]):
    pixel_labels.append("pixel"+str(i))
pixel_labels[:10]
print(resized_shape[1])
total_images=DF_Full.shape[0]
# Creating 1D array for all images
pixel_data=[]
num=0
for i in range(DF_Full.shape[0]):
    patientId=DF_Full.iloc[i]['patientId']
    dcm_file= '/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images/%s.dcm' % patientId
    dcm_data=pydicom.read_file(dcm_file)
    image=dcm_data.pixel_array
    
    final_pixel_array=[]
    for j in resize(image,resized_shape):
        final_pixel_array.extend(j)
    pixel_data.append(final_pixel_array)
    num=num+1
    if num==total_images:
        break
X=pd.DataFrame(data=pixel_data,columns=pixel_labels)
y=DF_Full['Target']
X.shape,y.shape
X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,random_state=42)
y_train_c = to_categorical(y_train)
y_test_c = to_categorical(y_test)
X_train_re = X_train.values.reshape(X_train.shape[0], resized_shape[0], resized_shape[1], 1)
X_test_re = X_test.values.reshape(X_test.shape[0], resized_shape[0], resized_shape[1], 1)
model = Sequential()

# First Convolution  Layer 
model.add(Conv2D(filters = 6,
                 kernel_size = 3,
                 activation = 'relu',
                 input_shape = (resized_shape[0], resized_shape[1], 1)
                ))

# Adding pooling
model.add(MaxPooling2D(pool_size=(2,2)))

# Second convolutional layer
model.add(Conv2D(filters=16,
                 kernel_size=3,
                 activation='relu'))

# Adding Pooling
model.add(MaxPooling2D(pool_size=(2,2)))

# Dropout
model.add(Dropout(0.5))

# Flatten
model.add(Flatten())

#Third Convoltion Layer
model.add(Dense(512,
                activation='relu'))

# Fourth Convoltion layer
model.add(Dense(128,
                activation='relu'))
# Dropout
model.add(Dropout(0.5))

#Output Layer
model.add(Dense(2, activation='softmax'))
model.summary()
model.compile(loss = 'categorical_crossentropy', 
              optimizer = 'adam', 
              metrics = ['accuracy'])
trained_model = model.fit(X_train_re,
                          y_train_c,
                          batch_size = 32,
                          validation_data = (X_test_re, y_test_c),
                          epochs = 20,
                          verbose = 1)
