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

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from PIL import Image as PImage
import cv2
df = pd.read_csv("/kaggle/input/coronahack-chest-xraydataset/Chest_xray_Corona_Metadata.csv")
training_images = ("/kaggle/input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train")
replace_dict = {'Pnemonia':1,
               'Normal':0}
df['Label'] = df['Label'].replace(replace_dict)
df = df.drop('Label_2_Virus_category', axis='columns')
df = df.drop('Label_1_Virus_category', axis='columns')
df = df.drop('Unnamed: 0', axis='columns')
training_labels = df[df.Dataset_type=='TRAIN']
testing_labels = df[df.Dataset_type=='TEST']
training_labels = training_labels.drop('Dataset_type', axis='columns')
testing_labels = testing_labels.drop('Dataset_type', axis='columns')
training_labels = training_labels.sort_values(by='X_ray_image_name') 

imageList = os.listdir(training_images)
imageList = sorted(imageList)
loadedImages = []
for images in imageList:
    if images in training_labels.values:
        img = PImage.open(training_images+'/'+images)
        img = img.resize((64, 64))
        pix_val = np.asarray(img)
         
        pix_val = pix_val/255
        if pix_val.shape != (64, 64):
            pix_val.reshape(4096,1)
            print(images, pix_val.shape)
            
        loadedImages.append(pix_val)
#1-s2.0-S1684118220300682-main.pdf-002-a1.png
    
df = pd.read_csv("/kaggle/input/coronahack-chest-xraydataset/Chest_xray_Corona_Metadata.csv")
training_images = ("/kaggle/input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train")
testing_images = ("/kaggle/input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test")
replace_dict = {'Pnemonia':1,
               'Normal':0}
df['Label'] = df['Label'].replace(replace_dict)
df = df.drop('Label_2_Virus_category', axis='columns')
df = df.drop('Label_1_Virus_category', axis='columns')
df = df.drop('Unnamed: 0', axis='columns')
training_labels = df[df.Dataset_type=='TRAIN']
testing_labels = df[df.Dataset_type=='TEST']
training_labels = training_labels.drop('Dataset_type', axis='columns')
testing_labels = testing_labels.drop('Dataset_type', axis='columns')
training_labels = training_labels.sort_values(by='X_ray_image_name') 
testing_labels = training_labels.sort_values(by='X_ray_image_name') 
training_labels.reset_index(drop=True, inplace=True)

        
#1-s2.0-S1684118220300682-main.pdf-002-a1.png
imageList = os.listdir(training_images)
imageList = sorted(imageList)
loadedImages = []
for images in imageList:
    if images in training_labels.values:
        img = cv2.imread(training_images+'/'+images)
        img = cv2.resize(img, (200,200))
        img = img/255    
        loadedImages.append(img)
        
        

        
realImage = "../input/chestexray1/chestxrayTest.jpg"
img = cv2.imread(realImage)
img = cv2.resize(img, (200,200))
img = img/255
realImage = img
realImageList = []
plt.figure()
plt.imshow(realImage)
realImageList.append(img)
realImageList = np.array(realImageList)
realImageList.shape
realLabels = [1]
plt.figure()
plt.imshow(loadedImages[1003])
loadedImages = np.array(loadedImages)

training_labels.drop('X_ray_image_name', axis='columns', inplace=True)


model = keras.Sequential()
model.add(keras.layers.Conv2D(filters=60, kernel_size = (3,3),
                              activation='relu',
                              input_shape=(200,200,3)))
model.add(keras.layers.Conv2D(filters=60, kernel_size = (3,3),
                              activation='relu',
                              input_shape=(200,200,3)))
model.add(keras.layers.MaxPool2D(pool_size = 2, strides=2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(2, activation='softmax'))
model.compile(optimizer='adam', #How you want to use gradient descent 
              loss='sparse_categorical_crossentropy', #loss function
              metrics=['accuracy'] #display accuracy while training
              )
train_labels = []
for i in range(len(training_labels)):
    train_labels.append(training_labels['Label'][i])
'''test_labels = []
for i in range(len(testing_labels)):
    test_label.append(testing_lables['Label'][i])'''
train_labels = np.array(train_labels)
'''test_labels = np.array(test_label)
print(type(train_labels))'''
model.fit(loadedImages[:2000], train_labels[:2000], epochs=1)
loss, accuracy = model.evaluate(np.array(loadedImages)[2050:3000], training_labels[2050:3000])
prediction = model.predict(np.array(realImageList))
print(prediction)