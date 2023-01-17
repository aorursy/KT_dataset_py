# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import math

import numpy as np

import pandas as pd

import h5py

import matplotlib.pyplot as plt

import scipy

from PIL import Image

from scipy import ndimage

import tensorflow as tf

from tensorflow.python.framework import ops



%matplotlib inline

np.random.seed(1)



def load_dataset():

    percent = 1.0

    train_set = pd.read_csv("../input/train.csv")

    test_set = pd.read_csv("../input/test.csv")

        

    Y_train = train_set["label"].values

    X_train = train_set.drop(["label"], axis=1) 



    X_test = test_set

    Y_test = None

    

    return X_train, Y_train, X_test, Y_test, 10

    

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()



img = np.array(X_train_orig.loc[1]).reshape(28,28)

plt.imshow(img)
X_train = (X_train_orig/255.).values.reshape(X_train_orig.shape[0], 28,28,1)

X_test = X_test_orig/255.

Y_train = Y_train_orig



import tensorflow as tf

import tensorflow.keras.layers as klayer

import tensorflow.keras.models as kmodel



model = tf.keras.models.Sequential([

    klayer.Conv2D(64, (3,3), activation='relu'),

    klayer.MaxPooling2D(2,2),

    klayer.Conv2D(32, (3,3), activation='relu'),

    klayer.MaxPooling2D(2,2),    

    klayer.Flatten(),

    klayer.Dense(256, activation="relu"),

    klayer.Dense(10, activation="softmax")

]) 
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=10)
model.summary()

Y_precit = tf.argmax(model.predict(X_test.values.reshape(X_test.shape[0], 28,28,1)), 1)
with tf.Session() as sess:

    z = sess.run(Y_precit)

    

submission_set = pd.read_csv("../input/sample_submission.csv")



for index, row in submission_set.iterrows():

    row['Label'] = z[index]

    

submission_set.to_csv('submission_set_CNN.csv', index=False)