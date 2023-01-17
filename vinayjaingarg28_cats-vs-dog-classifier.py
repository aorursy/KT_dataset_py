# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from tensorflow.keras.callbacks import TensorBoard
from keras import Sequential
from keras.layers import Dense,MaxPooling2D,Conv2D,Flatten,Dropout, BatchNormalization
import keras
from zipfile import ZipFile as zp

zip_file = '../input/train.zip'

with zp(zip_file, 'r') as zip_open:
    #zip_open.printdir()
    zip_open.extractall()
    
print('Done')
import cv2
from tqdm import tqdm
import h5py
path="./train"
label=[]
data1=[]
counter=0
img_size = 96

for file in os.listdir(path):
    image_data=cv2.imread(os.path.join(path,file), 0)  
    image_data=cv2.resize(image_data,(img_size,img_size))
    if file.startswith("cat"):
        label.append(0)
    elif file.startswith("dog"):
        label.append(1)
    try:
        data1.append(image_data/255)
    except:
        label=label[:len(label)-1]
    counter+=1
    if counter%1000==0:
        print (counter," image data retreived")
data1=np.array(data1)
print (data1.shape)
data1=data1.reshape((data1.shape)[0],(data1.shape)[1],(data1.shape)[2],1)
#data1=data1/255
labels=np.array(label)
from keras.optimizers import Adam, SGD
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=((data1.shape)[1],(data1.shape)[2],1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))

model.add(Conv2D(32, (3,3), activation='relu'))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
#model.add(Dropout(0.4))
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
model.summary()
model.fit(data1, label, epochs = 50, validation_split=0.1)
import matplotlib.pyplot as plt
history = model.history.history

plt.plot(history['loss'], label='Train')
plt.plot(history['val_loss'], label='Val')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper left')
plt.show()