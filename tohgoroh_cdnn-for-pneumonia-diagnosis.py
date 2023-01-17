# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import glob

from keras.preprocessing.image import load_img, img_to_array



img_size = (224, 224)

img_array_list = []

cls_list = []



dir = '../input/train/PNEUMONIA'

img_list = glob.glob(dir + '/*.jpeg')

for i in img_list:

    img = load_img(i, color_mode='grayscale', target_size=(img_size))

    img_array = img_to_array(img) / 255

    img_array_list.append(img_array)

    cls_list.append(1)



dir = '../input/train/NORMAL'

img_list = glob.glob(dir + '/*.jpeg')

for i in img_list:

    img = load_img(i, color_mode='grayscale', target_size=(img_size))

    img_array = img_to_array(img) / 255

    img_array_list.append(img_array)

    cls_list.append(0)



X_train = np.array(img_array_list)

y_train = np.array(cls_list)

print(X_train.shape, y_train.shape)
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import RandomOverSampler

from keras.utils.np_utils import to_categorical



X_learn, X_valid, y_learn, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=0)

shape = X_learn.shape



X_learn = np.reshape(X_learn, (shape[0], shape[1] * shape[2] * shape[3]))

ros = RandomOverSampler(random_state=0)

X_res, y_res = ros.fit_resample(X_learn, y_learn)

X_res = np.reshape(X_res, (X_res.shape[0], shape[1], shape[2], shape[3]))

Y_res = to_categorical(y_res)

Y_valid = to_categorical(y_valid)

print('Train:', X_res.shape, Y_res.shape)

print('Test:', X_valid.shape, Y_valid.shape)
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

import tensorflow as tf



def auc(y_true, y_pred):

    auc = tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

    return auc



model = Sequential()

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 1)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Dropout(rate=0.25))



model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 1)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Dropout(rate=0.25))



model.add(Flatten())

model.add(Dense(units=128, activation='relu'))

model.add(Dropout(rate=0.25))



model.add(Dense(units=2, activation='softmax'))



model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', auc])
model.fit(X_res, Y_res, batch_size=32, epochs=10)

model.evaluate(X_valid, Y_valid, batch_size=32)
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import RandomOverSampler

from keras.utils.np_utils import to_categorical



shape = X_train.shape



X_train = np.reshape(X_train, (shape[0], shape[1] * shape[2] * shape[3]))

ros = RandomOverSampler(random_state=0)

X_res, y_res = ros.fit_resample(X_train, y_train)

X_train = np.reshape(X_train, (shape[0], shape[1], shape[2], shape[3]))

X_res = np.reshape(X_res, (X_res.shape[0], shape[1], shape[2], shape[3]))

Y_res = to_categorical(y_res)

print('Train:', X_res.shape, Y_res.shape)
model.fit(X_res, Y_res, batch_size=32, epochs=10)
import glob

from keras.preprocessing.image import load_img, img_to_array



img_array_list = []



dir = '../input/test'

img_list = glob.glob(dir + '/*.jpeg')

img_list.sort()

for i in img_list:

    img = load_img(i, color_mode='grayscale', target_size=(img_size))

    img_array = img_to_array(img) / 255

    img_array_list.append(img_array)



X_test = np.array(img_array_list)

print(X_test.shape)
predict = model.predict(X_test)[:, 1]
submit = pd.read_csv('../input/sampleSubmission.csv')

submit['pneumonia'] = predict

submit.to_csv('submission.csv', index=False)