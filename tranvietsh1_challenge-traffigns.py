# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import glob

import keras

import matplotlib.pyplot as plt

import cv2

import numpy as np

from skimage import color, exposure, transform

import tensorflow as tf

from skimage import io

import os

import glob



print(os.listdir("../input/gtsrb_challenge/GTSRB_Challenge/train"))



# Any results you write to the current directory are saved as output.
NUM_CLASSES = 43

IMG_SIZE = 48
def preprocessing(img):

    return transform.resize(img, (IMG_SIZE,IMG_SIZE))



def get_class(img_path):

    return int(img_path.split('/')[-2])



root_dir = '../input/gtsrb_challenge/GTSRB_Challenge/train'

imgs = []

labels = []



all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))

np.random.shuffle(all_img_paths)

for img_path in all_img_paths:

    img = preprocessing(io.imread(img_path))

    label = get_class(img_path)

    imgs.append(img)

    labels.append(label)



X = np.array(imgs, dtype='float32')

# Make one hot targets

Y = np.eye(NUM_CLASSES, dtype='uint8')[labels]
def nameImage(img_path):

    return img_path.split('/')[-2]

fig=plt.figure(figsize=(20, 20))

columns = 9

rows = 9

for i in range(1, columns*rows +1):

    k = np.random.randint(0,len(imgs))

    fig.add_subplot(rows, columns, i)

    image = io.imread(all_img_paths[k])

    image = preprocessing(image)

    plt.imshow(image)

    plt.xticks([])

    plt.yticks([])

    plt.title(nameImage(all_img_paths[k]))

plt.show()
from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation, Flatten

from keras.layers.convolutional import Conv2D

from keras.layers.pooling import MaxPooling2D

from keras.optimizers import SGD

from keras import backend as K

K.set_image_data_format('channels_last')





def cnn_model():

    network = Sequential()

    network.add(Conv2D(64, (3, 3), padding='same',input_shape=(IMG_SIZE, IMG_SIZE,3),activation='relu'))

    network.add(Conv2D(64, (3, 3), activation='relu'))

    network.add(MaxPooling2D(pool_size=(2, 2)))

    network.add(Dropout(0.2))

    network.add(Conv2D(128, (3, 3), padding='same',activation='relu'))

    network.add(Conv2D(128, (3, 3), activation='relu'))

    network.add(MaxPooling2D(pool_size=(2, 2)))

    network.add(Dropout(0.2))

    network.add(Conv2D(512, (3, 3), padding='same',activation='relu'))

    network.add(Conv2D(512, (3, 3), activation='relu'))

    network.add(MaxPooling2D(pool_size=(2, 2)))

    network.add(Dropout(0.2))

    network.add(Flatten())

    network.add(Dense(1024, activation='relu'))

    network.add(Dropout(0.5))

    network.add(Dense(NUM_CLASSES, activation='softmax'))

    return network
from keras.optimizers import SGD

network = cnn_model()

lr = 0.01

sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)

network.compile(loss='categorical_crossentropy',

              optimizer=sgd,

              metrics=['accuracy'])
from keras.callbacks import LearningRateScheduler, ModelCheckpoint





def lr_schedule(epoch):

    return lr * (0.1 ** int(epoch / 10))



batch_size = 32

epochs = 30



network.fit(X, Y,

          batch_size=batch_size,

          epochs=epochs,

          validation_split=0.2,

          callbacks=[LearningRateScheduler(lr_schedule),

                     ModelCheckpoint('model_cnn.h5', save_best_only=True)]

          )
network.summary()
from keras.preprocessing.image import ImageDataGenerator

#from sklearn.cross_validation import train_test_split

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator



X_train, X_val, Y_train, Y_val = train_test_split(X, Y,

                                                  test_size=0.2, random_state=42)



datagen = ImageDataGenerator(featurewise_center=False,

                             featurewise_std_normalization=False,

                             width_shift_range=0.1,

                             height_shift_range=0.1,

                             zoom_range=0.2,

                             shear_range=0.1,

                             rotation_range=10.)



datagen.fit(X_train)



# Reinitialize model and compile

network = cnn_model()

network.compile(loss='categorical_crossentropy',

              optimizer=sgd,

              metrics=['accuracy'])



# Train again

epochs = 5

network.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),

                    steps_per_epoch=X_train.shape[0],

                    epochs=epochs,

                    validation_data=(X_val, Y_val),

                    callbacks=[LearningRateScheduler(lr_schedule),

                               ModelCheckpoint('model_T_A.h5', save_best_only=True)]

                    )
network.summary()


from keras.models import load_model

model_cnn = load_model('model_cnn.h5')



def get_name(img_path):

    return img_path.split('/')[-1]



import csv

with open('submission.csv', mode='w') as f:

    rowname = ['Filename', 'ClassId']

    writer = csv.DictWriter(f, fieldnames=rowname)

    writer.writeheader()



    root_dir = '../input/gtsrb_challenge/GTSRB_Challenge'

    all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))

    np.random.shuffle(all_img_paths)

    for img_path in all_img_paths:

        img = preprocessing(io.imread(img_path))

        name = get_name(img_path)

        y_pred = model_cnn.predict_classes(img.reshape(1,48,48,3))

        writer.writerow({'Filename': name, 'ClassId': int(y_pred)})

        #print(name + str(y_pred))
from keras.models import load_model

model_T_A = load_model('model_T_A.h5')



def get_name(img_path):

    return img_path.split('/')[-1]



import csv

with open('submission1.csv', mode='w') as f:

    rowname = ['Filename', 'ClassId']

    writer = csv.DictWriter(f, fieldnames=rowname)

    writer.writeheader()



    root_dir = '../input/gtsrb_challenge/GTSRB_Challenge'

    all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))

    np.random.shuffle(all_img_paths)

    for img_path in all_img_paths:

        img = preprocessing(io.imread(img_path))

        name = get_name(img_path)

        y_pred = model_T_A.predict_classes(img.reshape(1,48,48,3))

        writer.writerow({'Filename': name, 'ClassId': int(y_pred)})

        #print(name + str(y_pred))
fig=plt.figure(figsize=(20, 20))

columns = 9

rows = 9

root_dir = '../input/gtsrb_challenge/GTSRB_Challenge'

all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))

for i in range(1, columns*rows +1):

    k = np.random.randint(0,len(all_img_paths))

    fig.add_subplot(rows, columns, i)

    image = io.imread(all_img_paths[k])

    image = preprocessing(image)

    plt.imshow(image)

    plt.xticks([])

    plt.yticks([])

    y_pred_v = model_cnn.predict_classes(image.reshape(1,48,48,3))

    plt.title(int(y_pred_v))

plt.show()