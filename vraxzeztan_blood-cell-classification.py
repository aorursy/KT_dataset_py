# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/dataset2-master/dataset2-master/images"))

# Any results you write to the current directory are saved as output.
import os
import zipfile
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
%matplotlib inline
b_dir = "../input/dataset2-master/dataset2-master/images/"
def load_images(folder):
  dir = b_dir + folder
  images = []
  for folders in tqdm(os.listdir(dir)):
    for files in os.listdir(os.path.join(os.path.join(dir,folders))):
      #print(files,folders)
      filepath = os.path.join(os.path.join(dir,folders),files)
      img = cv2.imread(filepath)
      img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
      img = cv2.resize(img,(32,24))
      #img = np.asarray(img)
      img = img/255.
      images.append((img,folders))
  return images
train = load_images('TRAIN')
test = load_images('TEST')
val = load_images('TEST_SIMPLE')
def create_dataset(data):
  x = []
  y = []
  for obj in data:
    x.append(obj[0])
    y.append(obj[1])
  x = np.array(x)
  y = np.array(y)
  return x,y
x_train,y_train = create_dataset(train)
x_test,y_test = create_dataset(test)
x_val,y_val = create_dataset(val)
del train
del val
del test
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y_enc_train = encoder.fit_transform(y_train)
y_enc_test = encoder.transform(y_test)
y_enc_val = encoder.transform(y_val)
from keras.utils import to_categorical
y_cat_train = to_categorical(y_enc_train,4)
y_cat_test = to_categorical(y_enc_test,4)
y_cat_val = to_categorical(y_enc_val,4)
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
batch_size = 128
epochs = 100

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(24,32,3),strides=2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
model.fit(x_train,y_cat_train,batch_size,epochs ,validation_data=(x_test,y_cat_test))
