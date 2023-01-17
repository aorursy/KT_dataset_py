# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from os import listdir, makedirs
from os.path import join, exists, expanduser
import cv2
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense,BatchNormalization
from keras import backend as K
import tensorflow as tf
from scipy import io
from keras.utils import np_utils
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_data = io.loadmat('/kaggle/input/petslist/PetsTrain.mat')
test_data = io.loadmat('/kaggle/input/petslist/PetsTest.mat')

train_images = []
train_labels = []

test_images = []
test_labels = []

for i in range(len(train_data['files'])):
    tempImage = cv2.imread('/kaggle/input/the-oxfordiiit-pet-dataset/images/' + train_data['files'][i][0][0])
    if tempImage is not None:
        tempImage = cv2.normalize(tempImage, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        image_resized = cv2.resize(tempImage, (224,224), interpolation = cv2.INTER_AREA)
        train_images += [image_resized]
        train_labels += [train_data['label'][i]]
        
for i in range(len(test_data['files'])):
    
    tempImage = cv2.imread('/kaggle/input/the-oxfordiiit-pet-dataset/images/' + test_data['files'][i][0][0])
    if tempImage is not None:
        tempImage = cv2.normalize(tempImage, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        tempImage = cv2.resize(tempImage, (224,224), interpolation = cv2.INTER_AREA)
        test_images += [tempImage]
        test_labels += [test_data['label'][i]]
num_classes = 38
train_labels = np_utils.to_categorical(train_labels,num_classes)
test_labels = np_utils.to_categorical(test_labels,num_classes)
val_images = train_images[-1000:]
val_labels = train_labels[-1000:]
train_images = train_images[:-1000]
train_labels = train_labels[:-1000]
np.shape(val_images)
batch_size = 64

ResNet_base = applications.ResNet50(weights='imagenet',include_top=False,input_shape= (224,224,3))
x = ResNet_base.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = BatchNormalization()(x)
predictions = Dense(num_classes, activation='softmax')(x)
ResNet_transfer = Model(inputs=ResNet_base.input, outputs=predictions)
ResNet_transfer.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(),
              metrics=['accuracy'])
ResNet_transfer.summary()
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
print('# Fit model on training data')
import tensorflow as tf
with tf.device("/device:GPU:0"):
    history = ResNet_transfer.fit(np.array(train_images), np.array(train_labels),
                        batch_size=batch_size,
                        epochs=25,
                        validation_data=(np.array(val_images), np.array(val_labels)))

    print('\nhistory dict:', history.history)
print('\n# Evaluate on test data')
results = ResNet_transfer.evaluate(np.array(test_images),np.array(test_labels),batch_size=batch_size)
print('test loss, test acc:', results)
