# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import glob

import random

import base64

from PIL import Image

from io import BytesIO

from IPython.display import HTML

import tensorflow as tf

from tensorflow import keras

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt

import os

print(os.listdir("../input"))

from subprocess import check_output

print(check_output(["ls", "../input/self driving car training data/data"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
d=pd.read_csv('../input/self driving car training data/data/driving_log.csv')

d.head()
pd.set_option('display.max_colwidth', -1)



def get_thumbnail(path):

    i = Image.open(path)

    i.thumbnail((150, 150), Image.LANCZOS)

    return i



def image_base64(im):

    if isinstance(im, str):

        im = get_thumbnail(im)

    with BytesIO() as buffer:

        im.save(buffer, 'jpeg')

        return base64.b64encode(buffer.getvalue()).decode()



def image_formatter(im):

    return f'<img src="data:image/jpeg;base64,{image_base64(im)}">'
d['file'] = d.center.map(lambda id: f'../input/self driving car training data/data/{id}')

d['image'] = d.file.map(lambda f: get_thumbnail(f))

HTML(d[['speed', 'image']][1:10].to_html(formatters={'image': image_formatter}, escape=False))
x=d.pop('image')

y=d.pop('file')

train_dataset = d.sample(frac=0.7,random_state=0)

test_dataset = d.drop(train_dataset.index)

train_y=train_dataset[['speed','throttle','steering','brake']].copy()

train_dataset.drop(['speed','throttle','steering','brake'], axis=1, inplace=True)

test_y=test_dataset[['speed','throttle','steering','brake']].copy()

test_dataset.drop(['speed','throttle','steering','brake'], axis=1, inplace=True)

                  









def path_to_tensor(dset):

    img=[]

    for img_path in dset.itertuples():

        xx=img_path[1]

        xx.replace(" ", "")

        imgx = Image.open('../input/self driving car training data/data/'+xx)

        imgxx=np.asarray(imgx,dtype='int32')

        img.append(imgxx)

    return np.array(img)

train=path_to_tensor(train_dataset)

test=path_to_tensor(test_dataset)

xtrain = train.reshape(train.shape[0], 160, 320, 3)

ytrain=test.reshape(test.shape[0], 160, 320, 3)
print(xtrain.shape,ytrain.shape)
input_shape = (160,320,3)

model=keras.Sequential()

model.add(keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1),

                 activation='relu',

                 input_shape=input_shape))

model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(keras.layers.Conv2D(64, (5, 5), activation='relu'))

model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(1000, activation='relu'))

model.add(keras.layers.Dense(4))



optimizer = keras.optimizers.RMSprop(0.001)



model.compile(loss='mean_squared_error',

            optimizer=optimizer,

            metrics=['mean_absolute_error', 'mean_squared_error'])



    
model.summary()
EPOCHS=20

history = model.fit(

  xtrain, train_y,batch_size=64,

  epochs=EPOCHS, validation_split = 0.2, verbose=1)
pre2=model.evaluate(ytrain,test_y,batch_size = 64)
def plot_history(history):

  hist = pd.DataFrame(history.history)

  hist['epoch'] = history.epoch

  

  plt.figure()

  plt.xlabel('Epoch')

  plt.ylabel('Mean Abs Error [MPG]')

  plt.plot(hist['epoch'], hist['mean_absolute_error'],

           label='Train Error')

  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],

           label = 'Val Error')

  plt.ylim([0,100])

  plt.legend()

  

  plt.figure()

  plt.xlabel('Epoch')

  plt.ylabel('Mean Square Error [$MPG^2$]')

  plt.plot(hist['epoch'], hist['mean_squared_error'],

           label='Train Error')

  plt.plot(hist['epoch'], hist['val_mean_squared_error'],

           label = 'Val Error')

  plt.ylim([0,100])

  plt.legend()

  plt.show()





plot_history(history)

print("Accuracy of model on test data is: ",pre2[1]*100)
y=model.predict(ytrain[:,:,:,:])
