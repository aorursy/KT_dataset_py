# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

       # print(os.path.join(dirname, filename))

        pass



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/coronahack-chest-xraydataset/Chest_xray_Corona_Metadata.csv')

data = data[['X_ray_image_name','Label','Dataset_type']]

data = data.dropna()

data
from tqdm import tqdm



data['File Path'] = 0

base_string = "/kaggle/input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/"

for index in tqdm(data.index):

    if data.loc[index,'Dataset_type'] == 'TRAIN':

        string = base_string + 'train/'

    else:

        string = base_string + 'test/'

    data.loc[index,'File Path'] = string + data.loc[index,'X_ray_image_name']
import matplotlib.pyplot as plt

import time

import cv2

X = []

for image in tqdm(data['File Path']):

    X.append(cv2.resize(cv2.imread(image),(100,100)))
X = np.array(X)
data['Label'] = data['Label'].map({'Normal':0,'Pnemonia':1})
y = data['Label']

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
from keras.applications.resnet import ResNet50

from keras.layers import Flatten,Dense,Dropout,Activation

from keras.models import Sequential

model = Sequential()

model.add(ResNet50(input_shape=(100,100,3),classes=1,include_top=False))

model.add(Flatten())

model.add(Dense(10,activation='relu'))

model.add(Activation('relu'))

model.add(Dense(25,activation='relu'))

model.add(Activation('relu'))

model.add(Dense(1,activation='relu'))

model.add(Activation('sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])



model.summary()
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(

    featurewise_center=True,

    featurewise_std_normalization=True,

    rotation_range=20,

    width_shift_range=0.2,

    height_shift_range=0.2,

    horizontal_flip=True)

datagen.fit(X_train)
#epochs = 5

#model.fit_generator(datagen.flow(X_train, y_train, batch_size=128),

#                    steps_per_epoch=len(X_train)/128, epochs=epochs)

import time

start = time.time()

model.fit(x=X_train,y=y_train,validation_data=(X_test,y_test),epochs = 30)

end = time.time()

print("TRAINING TIME:",end-start)
model.save_weights('weights.h5')