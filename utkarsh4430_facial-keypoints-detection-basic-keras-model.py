#checking where the files are

import os

os.listdir("/kaggle/input/training")
#importing the rquired libraries

import pandas as pd

import numpy as np

import keras

from tqdm import tqdm

from keras import backend as K
lookid_data = pd.read_csv("/kaggle/input/IdLookupTable.csv")

lookid_data.head()
samplesubmission = pd.read_csv("/kaggle/input/SampleSubmission.csv")

samplesubmission.head()
train = pd.read_csv("/kaggle/input/training/training.csv")

train.head().T
train.isnull().sum()
#filling the nan values

train.fillna(method = 'ffill',inplace = True)
X = train.Image.values

del train['Image']

Y = train.values
x = []

for i in tqdm(X):

    q = [int(j) for j in i.split()]

    x.append(q)

len(x)
x = np.array(x)

x = x.reshape(7049, 96,96,1)

x  = x/255.0

x.shape
from sklearn.model_selection import train_test_split as tts

x_train,x_test,y_train,y_test = tts(x,Y,random_state = 69,test_size = 0.1)
x_train.shape,x_test.shape,y_train.shape,y_test.shape
from keras.layers.advanced_activations import LeakyReLU

from keras.models import Sequential, Model

from keras.layers import Activation, Convolution2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Conv2D,MaxPool2D, ZeroPadding2D
model = Sequential()



model.add(Convolution2D(32, (3,3), padding='same', use_bias=False, input_shape=(96,96,1)))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())



model.add(Convolution2D(32, (3,3), padding='same', use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())



model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())



model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))

# model.add(BatchNormalization())

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())



model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())



model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())



model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())





model.add(Flatten())

model.add(Dense(512,activation='relu'))

model.add(Dropout(0.1))

model.add(Dense(30))

model.summary()
# def root_mean_squared_error(y_true, y_pred):

#         return K.sqrt(K.mean(K.square(y_pred - y_true)))
model.compile(optimizer = 'adam',loss = 'mean_squared_error', metrics = ['mae','acc'])

model.fit(x_train,y_train,batch_size=256, epochs=50,validation_data=(x_test,y_test))
model.compile(optimizer = 'adam',loss = 'mean_squared_error', metrics = ['mae'])

model.fit(x,Y,batch_size=64, epochs=100)

model.fit(x,Y,batch_size=128, epochs=50)

model.fit(x,Y,batch_size=256, epochs=50)
test = pd.read_csv("/kaggle/input/test/test.csv")

test.head()
test.isnull().sum()
test = test.Image.values

x_t = []

for i in tqdm(test):

    q = [int(j) for j in i.split()]

    x_t.append(q)

x_t = np.array(x_t)

x_t = x_t.reshape(-1, 96,96,1)

x_t = x_t/255.0

x_t.shape
pred = model.predict(x_t)

pred.shape
lookid_list = list(lookid_data['FeatureName'])

imageID = list(lookid_data['ImageId']-1)

pre_list = list(pred)
rowid = lookid_data['RowId']

rowid=list(rowid)
feature = []

for f in list(lookid_data['FeatureName']):

    feature.append(lookid_list.index(f))
preded = []

for x,y in zip(imageID,feature):

    preded.append(pre_list[x][y])
rowid = pd.Series(rowid,name = 'RowId')
loc = pd.Series(preded,name = 'Location')
submission = pd.concat([rowid,loc],axis = 1)
submission.to_csv('Utkarsh.csv',index = False)