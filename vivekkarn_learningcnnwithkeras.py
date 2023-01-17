# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
from glob import glob
import random
import cv2
import matplotlib.pylab as plt
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#opening csv files with labels
csv = pd.read_csv('../input/sample_labels.csv')
csv.head()
#listing all the images in the directory
images_list = glob(os.path.join('..', 'input', 'sample', '*', '*.png'))
#using only 500 images for computaional reasons
images = images_list[0:1000]
#creating arrays of labels and images for training the network
labels = []
imgs = []
for image in images:
    finding = csv.loc[csv['Image Index'] == os.path.basename(image),'Finding Labels'].values[0]
    if finding in 'No Finding':
        label = 1
    else:
        label = 0
    labels.append(label)
    img = cv2.imread(image)
    img = cv2.resize(img, (512,512))
    #img = img.reshape(786432)
    imgs.append(img)
#converting into numpy array so that it can easily be fed into the keras model
data = np.array(imgs)
labels = np.array(labels)
#splitting the training data into train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, random_state=42)
#defining the model using keras
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Lambda, MaxPool2D, BatchNormalization, Input
from keras.models import Model
inputs = Input(shape=(512,512,3))
x = Flatten()(inputs)
x = Dense(512, activation='relu')(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)
#training the model
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=32, epochs=100)
#evaluting the model
model.evaluate( X_test, y_test)
#i am still learning now. more changes on coming days