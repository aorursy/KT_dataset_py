

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.preprocessing.image import img_to_array


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import glob
import cv2 as cv
import matplotlib.pyplot as plt
x_train=[]
y_train=[]
train_path='/kaggle/input/intel-image-classification/seg_train/seg_train'
for i in os.listdir(train_path):
    path=(os.path.join(train_path,i))
    files=glob.glob(path+'/*.jpg')
    for file in files:
        img=cv.imread(file)
        img=cv.resize(img,(64,64))
        x_train.append(list(img))
        y_train.append(i)

              

test_path='/kaggle/input/intel-image-classification/seg_test/seg_test'
x_test=[]
y_test=[]
for i in os.listdir(test_path):
    path=os.path.join(test_path,i)
    files=glob.glob(path+'/*.jpg')
    for file in files:
        img=cv.imread(file)
        img=cv.resize(img,(64,64))
        x_test.append(list(img))
        y_test.append(i)

x_test=np.array(x_test)
print(x_test.shape)

y_test=np.array(y_test)
print(y_test.shape)
pd.Series(y_test).value_counts()

plt.figure(figsize=(20,10))
for i,j in enumerate(np.random.randint(0,len(x_train),25)):
    plt.subplot(5,5,i+1)
    plt.imshow(x_train[j])
    plt.title(y_train[j])
    plt.grid(False)
    plt.xticks([])
len(y_train)
x_train=np.array(x_train)
y_train=np.array(y_train)
renames={'buildings':0,'glacier':1,'street':2,'sea':3,'forest':4,'mountain':5}
y_train=pd.Series(y_train).map(renames)
y_test=pd.Series(y_test).map(renames)
y_test.value_counts()
y_train.value_counts()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,MaxPooling2D,Flatten,Dropout,Conv2D
model=Sequential([])
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(64,64,3)))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(4,4))

model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(.1))
model.add(Dense(64,activation='relu'))
model.add(Dropout(.1))
model.add(Dense(6,activation='softmax'))

import tensorflow
losses=tensorflow.keras.losses.sparse_categorical_crossentropy
model.compile(optimizer='adam',loss=losses,metrics=['accuracy'])

history=model.fit(x_train,y_train,epochs=10)
acc=model.evaluate(x_test,y_test)