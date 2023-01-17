import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D, BatchNormalization



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





import zipfile



with zipfile.ZipFile("../input/dogs-vs-cats/train.zip","r") as z:

    z.extractall(".")
main_dir = "/kaggle/working/"

train_dir = "train"

path = os.path.join(main_dir,train_dir)



X = []

y = []

convert = lambda category : int(category == 'dog')

def create_dataset(path):

    for p in os.listdir(path):

        category = p.split(".")[0]

        category = convert(category)

        img_array = cv2.imread(os.path.join(path,p),cv2.IMREAD_GRAYSCALE)

        new_img_array = cv2.resize(img_array, dsize=(80, 80))

        X.append(new_img_array)

        y.append(category)
create_dataset(path)

X = np.array(X).reshape(-1, 80,80,1)

y = np.array(y)

X = X/255.0

print(X.shape)

print(y.shape)
model = Sequential()



model.add(Conv2D(64,(3,3), activation = 'relu', input_shape = X.shape[1:]))

model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Dropout(0.5))

model.add(Conv2D(64,(3,3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.summary()
model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=20, batch_size=128, validation_split=0.2)
X_train = np.array(X).reshape(-1, 6400)

y_train = np.array(y)

print(X_train.shape)

print(y_train.shape)
model_2 = Sequential()

model_2.add(Dense(4096, activation='relu', input_shape = X_train.shape[1:]))

model_2.add(Dropout(0.5))

model_2.add(Dense(2048, activation='relu'))

model_2.add(Dropout(0.5))

model_2.add(BatchNormalization())

model_2.add(Dense(512,activation='relu'))

model_2.add(Dropout(0.5))

model_2.add(Dense(256, activation='relu'))

model_2.add(Dropout(0.5))

model_2.add(BatchNormalization())

model_2.add(Dense(64, activation='relu'))

model_2.add(Dense(1,activation='sigmoid'))

model_2.summary()
model_2.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
model_2.fit(X_train, y_train, epochs=30, batch_size=128, validation_split=0.2)