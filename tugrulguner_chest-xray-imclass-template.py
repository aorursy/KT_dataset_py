# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow.keras as tf
import cv2
from ast import literal_eval
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import glob
from sklearn.metrics import accuracy_score

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def imreadandresffold(path, fileextension, sizeh, sizew, operation):
  if operation is 'all':
    imdata= np.stack([cv2.resize(cv2.imread(file)/np.max(cv2.imread(file)), (sizeh,sizew), interpolation = cv2.INTER_NEAREST) for file in glob.glob(path + '*.' + fileextension)])
  else:
    imdata= np.stack([cv2.resize(cv2.imread(file)/np.max(cv2.imread(file)), (sizeh,sizew), interpolation = cv2.INTER_NEAREST) for file in glob.glob(path + operation + '.' + fileextension)])
  return imdata

path = '/kaggle/input/chest-xray-pneumonia/chest_xray/train/NORMAL/'
pathpne = '/kaggle/input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/'
imdatanorm = imreadandresffold(path, 'jpeg', 128, 128, 'all')
imdatapne_bac = imreadandresffold(pathpne, 'jpeg', 128, 128, 'person*_bacteria_*')
imdatapne_vir = imreadandresffold(pathpne, 'jpeg', 128, 128, 'person*_virus_*')
imdatapne_vir.shape
plt.imshow(imdatapne_vir[0,...])
trainingdata = np.concatenate([imdatanorm, imdatapne_bac, imdatapne_vir])
label = np.zeros((trainingdata.shape[0],3))
label[0:imdatanorm.shape[0],0] = 1
label[imdatanorm.shape[0]:(imdatanorm.shape[0]+imdatapne_bac.shape[0]),1] = 1
label[(imdatanorm.shape[0]+imdatapne_bac.shape[0]):trainingdata.shape[0],2] = 1
indexes = np.arange(trainingdata.shape[0])
np.random.shuffle(indexes)
trainindatashuff = trainingdata[indexes,...]
labelshuff = label[indexes]
labelshuff = labelshuff.astype(int)
input_layer = tf.Input(shape=(128, 128, 3))

x = tf.layers.Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu')(input_layer)
#x = tf.layers.Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu')(x)
x = tf.layers.MaxPool2D((2, 2), padding='same', strides=(1, 1))(x)
x = tf.layers.Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu')(x)
#x = tf.layers.Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu')(x)
x = tf.layers.MaxPool2D((2, 2), padding='same', strides=(1, 1))(x)
x = tf.layers.Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu')(x)
x = tf.layers.MaxPool2D((2, 2), padding='same', strides=(1, 1))(x)
#x = tf.layers.Conv2D(64, (1, 1), padding='same', strides=(1, 1), activation='relu')(x)
#x = tf.layers.MaxPool2D((2, 2), padding='same', strides=(1, 1))(x)
#x = tf.layers.Conv2D(64, (1, 1), padding='same', strides=(1, 1), activation='relu')(x)
#x = tf.layers.Conv2D(64, (1, 1), padding='same', strides=(1, 1), activation='relu')(x)
#x = tf.layers.MaxPool2D((2, 2), padding='same', strides=(1, 1))(x)
#x = tf.layers.Lambda(lambda x: x**2)(x)
x = tf.layers.Flatten()(x)
x = tf.layers.Dense(128, activation='relu')(x)
x = tf.layers.Dense(128, activation='relu')(x)
x = tf.layers.Dense(3)(x)

initial_learning_rate = 0.01
lr_schedule = tf.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=10,
    decay_rate=0.96,
    staircase=True)
# you can use lr_schedule if you want your learning rate to decay, but I am not using it right now 
model = tf.Model(input_layer, x)
#losse = tf.losses.mse()
model.compile(loss='mse', optimizer=tf.optimizers.Adam())
model.fit(trainindatashuff[0:1000,...], labelshuff[0:1000,...], batch_size=50, epochs=10)
veri = trainindatashuff[1000:1500,...]
veri = veri.reshape(500,128,128,3)
predictions = model.predict(veri)
accuracy_score(np.argmax(predictions,axis=1),np.argmax(labelshuff[1000:1500,...],axis=1))
