# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import random
import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
%matplotlib inline
os.mkdir('train')
os.mkdir('train/cat')
os.mkdir('train/dog')

for i in os.listdir('../input/train'):
    if i[0] == 'c':
        os.symlink('../../../input/train/'+i, 'train/cat/'+i)
    else:
        os.symlink('../../../input/train/'+i, 'train/dog/'+i)
train_datagen = ImageDataGenerator(rescale=1.0/255,
                                  rotation_range=15,
                                  zoom_range=0.2,)

train_generator = train_datagen.flow_from_directory(
        'train',  # this is the target directory
        target_size=(224, 224),  # all images will be resized to 224x224
        batch_size=16,
        class_mode='binary')

x, y = train_generator.next()

plt.figure(figsize=(9, 9))
for i, (img, label) in enumerate(zip(x, y)):
    plt.subplot(4, 4, i+1)
    if label == 1:
        plt.title('dog')
    else:
        plt.title('cat')
    plt.axis('off')
    plt.imshow(img, interpolation="nearest")
from keras.applications.vgg16 import VGG16
base_model = VGG16(weights=None, include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1, activation='sigmoid', name='output')(x)

model = Model(input=base_model.input, output=x)
model.compile(loss='binary_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit_generator(
        train_generator,
        samples_per_epoch=16,
        nb_epoch=1)