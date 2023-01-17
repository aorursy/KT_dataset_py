# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import os

import glob

import tensorflow as tf

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn import metrics



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
def load_Data():

    train_path = "../input/train/"

    sub_folders = check_output(["ls", train_path]).decode("utf8").strip().split('\n')

    labels = sub_folders

    X_train = []

    X_train_id = []

    y_train = []





    print('Read train images')

    folders = sub_folders

    for fld in folders:

        index = folders.index(fld)

        print('Load folder {}'.format(fld, index))

        path = os.path.join('..', 'input', 'train', fld, '*.jpg')

        files = glob.glob(path)

        for fl in files:

            flbase = os.path.basename(fl)

            img = cv2.imread(fl)

            img = cv2.resize(img, (256, 256), cv2.INTER_LINEAR)

            X_train.append(img)

            X_train_id.append(flbase)

            y_train.append(index)



    return X_train, y_train, X_train_id





    

  
X_train, y_train, X_train_id = load_Data()

x = X_train

y =  y_train
y[3000]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=32)
X_train[0].shape[0]
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, X_train[0].shape[0]])

y_ = tf.placeholder(tf.float32, shape=[None, 10])



import tflearn

from tflearn.layers.core import input_data, dropout, fully_connected

from tflearn.layers.conv import conv_2d, max_pool_2d

from tflearn.layers.estimator import regression



network = input_data(shape=[None, 224, 224, 3])



network = conv_2d(network, 64, 3, activation='relu')

network = conv_2d(network, 64, 3, activation='relu')

network = max_pool_2d(network, 2, strides=2)



network = conv_2d(network, 128, 3, activation='relu')

network = conv_2d(network, 128, 3, activation='relu')

network = max_pool_2d(network, 2, strides=2)



network = conv_2d(network, 256, 3, activation='relu')

network = conv_2d(network, 256, 3, activation='relu')

network = conv_2d(network, 256, 3, activation='relu')

network = max_pool_2d(network, 2, strides=2)



network = conv_2d(network, 512, 3, activation='relu')

network = conv_2d(network, 512, 3, activation='relu')

network = conv_2d(network, 512, 3, activation='relu')

network = max_pool_2d(network, 2, strides=2)



network = conv_2d(network, 512, 3, activation='relu')

network = conv_2d(network, 512, 3, activation='relu')

network = conv_2d(network, 512, 3, activation='relu')

network = max_pool_2d(network, 2, strides=2)



network = fully_connected(network, 4096, activation='relu')

network = dropout(network, 0.5)

network = fully_connected(network, 4096, activation='relu')

network = dropout(network, 0.5)

network = fully_connected(network, 17, activation='softmax')



network = regression(network, optimizer='rmsprop',

                     loss='categorical_crossentropy',

                     learning_rate=0.001)



# Training

model = tflearn.DNN(network, checkpoint_path='model_vgg',

                    max_checkpoints=1, tensorboard_verbose=0)

model.fit(X_train, y_train, n_epoch=500, shuffle=True,

          show_metric=True, batch_size=32, snapshot_step=500,

          snapshot_epoch=False, run_id='vgg')