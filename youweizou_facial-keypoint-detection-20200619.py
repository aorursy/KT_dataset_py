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
import matplotlib.pyplot as plt
train = pd.read_csv("/kaggle/input/facial-keypoints-detection/training.zip")

test = pd.read_csv("/kaggle/input/facial-keypoints-detection/test.zip")

train.shape, test.shape
train.isnull().any().value_counts()
train.fillna(method='ffill', inplace=True)
imgs_train = train['Image'].apply(lambda x: x.split(' '))

imgs_test = test['Image'].apply(lambda x: x.split(' '))

y_train = []

for i in range(len(train)):

    y_train.append(list(train.iloc[i, :-1].values))

np.array(y_train).shape
def show_imgs(feature_list, imgs_list, n):

    imgs_pixel = np.array(imgs_list[n]).reshape(96, 96)

    for i in range(len(y_train[0])//2):

        x = feature_list[n][2*i]

        y = feature_list[n][2*i + 1]

        imgs_pixel[int(y), int(x)] = 255

    plt.imshow(imgs_pixel.astype(np.float))
X_train = []

X_test = []

for i in range(train.shape[0]):

    image = np.array(train['Image'][i].split(' '), dtype=int)

    X_train.append(np.reshape(image, (96, 96, 1)))

X_train = np.array(X_train)/255

for i in range(test.shape[0]):

    image = np.array(test['Image'][i].split(' '), dtype=int)

    X_test.append(np.reshape(image, (96, 96, 1)))

X_test = np.array(X_test)/255

# X_train = train['Image'].apply(lambda x: x.split(' '))

# X_train = np.array(X_train).reshape(-1, 96, 96, 1)

# X_test = np.array(X_test).reshape(-1, 96, 96, 1)
from keras.layers import Conv2D, Convolution2D, Dense, Flatten, Dropout, MaxPooling2D, BatchNormalization, MaxPool2D

from keras.models import Sequential, Model

from keras.layers.advanced_activations import LeakyReLU
model = Sequential()



model.add(Convolution2D(32, (3,3), padding='same', use_bias=False, input_shape=(96,96,1), activation='relu'))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())



model.add(Convolution2D(32, (3,3), padding='same', use_bias=False, activation='relu'))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Convolution2D(64, (3,3), padding='same', use_bias=False, activation='relu'))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())



model.add(Convolution2D(64, (3,3), padding='same', use_bias=False, activation='relu'))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Convolution2D(96, (3,3), padding='same', use_bias=False, activation='relu'))

# model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())



model.add(Convolution2D(96, (3,3), padding='same', use_bias=False, activation='relu'))

# model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Convolution2D(128, (3,3),padding='same', use_bias=False, activation='relu'))

# model.add(BatchNormalization())

# model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())



model.add(Convolution2D(128, (3,3),padding='same', use_bias=False, activation='relu'))

# model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Convolution2D(256, (3,3),padding='same',use_bias=False, activation='relu'))

# model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())



model.add(Convolution2D(256, (3,3),padding='same',use_bias=False, activation='relu'))

# model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Convolution2D(512, (3,3), padding='same', use_bias=False, activation='relu'))

# model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())



model.add(Convolution2D(512, (3,3), padding='same', use_bias=False, activation='relu'))

# model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())





model.add(Flatten())

model.add(Dense(512,activation='relu'))

model.add(Dropout(0.1))

model.add(Dense(30))

model.summary()
model.compile(optimizer='adam', 

              loss='mean_squared_error',

              metrics=['mae'])
model.fit(X_train, np.array(y_train),epochs = 50, batch_size=256, validation_split = 0.2)
pred = model.predict(X_test)
show_imgs(pred, imgs_test, 28)
IdLookup = pd.read_csv("/kaggle/input/facial-keypoints-detection/IdLookupTable.csv")
ImageId = IdLookup['ImageId']-1
Feature = list(IdLookup['FeatureName'][:30])

features = []

for feature in IdLookup['FeatureName']:

    features.append(Feature.index(feature))
loc = []

for image, name in zip(ImageId, features):

    loc.append(list(pred)[image][name])
sub = pd.read_csv("/kaggle/input/facial-keypoints-detection/SampleSubmission.csv")

sub['Location'] = loc

sub['Location'][sub['Location'] > 96] = 95

sub.to_csv('Sub-facial_detection.csv', index=False)