# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Importing Libraries

import matplotlib.pyplot as plt

import tensorflow

import cv2

import keras

from keras.models import Sequential

from keras.layers import Dense, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Cropping2D
# Importing Driving csv:

df = pd.read_csv('/kaggle/input/selfdriving-car-udacity/driving_log.csv',header=None)

df.head()
#Let's prepare the Dataset

from IPython.display import clear_output

images = []

angles = []

for i in range(df.shape[0]):

    clear_output(wait=True)

    im_c_path = '/kaggle/input/selfdriving-car-udacity/IMG/IMG/'+str(df[0][i].split('\\')[-1])

    im_l_path = '/kaggle/input/selfdriving-car-udacity/IMG/IMG/'+str(df[1][i].split('\\')[-1])

    im_r_path = '/kaggle/input/selfdriving-car-udacity/IMG/IMG/'+str(df[2][i].split('\\')[-1])

    print(f'Loading {i+1} image')

    print(im_c_path)

    im_c = cv2.imread(im_c_path)

    im_l = cv2.imread(im_l_path)

    im_r = cv2.imread(im_r_path)



    images.append(im_c)

    images.append(im_l)

    images.append(im_r)

    

    angles.append(df[3][i])

    angles.append(df[3][i])

    angles.append(df[3][i])

print('Done')
a = len(angles)

images = np.asarray(images)

angles = np.asarray(angles).reshape(a,1)

print('Done')
model = Sequential()

model.add(Conv2D(64, (5, 5), strides=(2, 2), activation='relu',input_shape=(160,320,3)))

model.add(Conv2D(32, (5, 5), strides=(2, 2), activation='relu'))

model.add(MaxPooling2D((2,2)))

model.add(Conv2D(32, (5, 5), strides=(2, 2), activation='relu'))





model.add(Conv2D(64, (3, 3), strides=(2, 2), activation='relu'))

model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dropout(0.2))

model.add(Dense(512))



model.add(Dense(256))

model.add(Dense(256))

model.add(Dense(1))

model.summary()
model.compile(loss='mse',optimizer='rmsprop')

history = model.fit(images,angles,epochs = 15,validation_split=0.2,shuffle=True)
model.save('model.h5')
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.show()