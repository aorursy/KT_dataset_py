# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir('../input/panoai2'))

# Any results you write to the current directory are saved as output.
# coding:utf-8
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, GlobalAveragePooling2D, \
    BatchNormalization
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import os
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras_preprocessing.image import ImageDataGenerator
from keras.engine.saving import load_model

shuffle = 10000
file = '../input/panoai2'

def analy_TFrecord(input):
    Features = {"id": tf.FixedLenFeature([], tf.int64),
                "data": tf.FixedLenFeature([256, 256], tf.float32),
                "label": tf.FixedLenFeature([], tf.int64)}
    Features = tf.parse_single_example(input, Features)
    return Features["id"], tf.one_hot(Features["label"] - 1, 5), Features["data"]


def train_load():
    path = [os.path.join(file, 'TFcodeX_{}.tfrecord'.format(i)) for i in range(1, 10)]
    my_dataset = tf.data.TFRecordDataset([path])
    my_dataset = my_dataset.map(analy_TFrecord)
    if shuffle:
        my_dataset = my_dataset.shuffle(buffer_size=shuffle)
    iterator = my_dataset.make_one_shot_iterator().get_next()
    labels = []
    X_test = []
    with tf.Session() as sess:
        while True:
            try:
                id, label_onehot, pic = sess.run(iterator)
                labels.append(label_onehot)
                X_test.append(pic)
            except:
                break
    return np.asarray(X_test).reshape(-1, 256, 256, 1), np.asarray(labels)


def test_load():
    path = [os.path.join(file, 'TFcodeX_{}.tfrecord'.format(i)) for i in range(10, 11)]
    my_dataset = tf.data.TFRecordDataset([path])
    my_dataset = my_dataset.map(analy_TFrecord)
    if shuffle:
        my_dataset = my_dataset.shuffle(buffer_size=shuffle)
    iterator = my_dataset.make_one_shot_iterator().get_next()
    labels = []
    X_test = []
    with tf.Session() as sess:
        while True:
            try:
                id, label_onehot, pic = sess.run(iterator)
                labels.append(label_onehot)
                X_test.append(pic)
            except:
                break
    return np.asarray(X_test).reshape(-1, 256, 256, 1), np.asarray(labels)

def test_load_office(path):
    my_dataset = tf.data.TFRecordDataset([path])
    my_dataset = my_dataset.map(analy_TFrecord)
    iterator = my_dataset.make_one_shot_iterator().get_next()
    labels = []
    X_test = []
    with tf.Session() as sess:
        while True:
            try:
                id, label_onehot, pic = sess.run(iterator)
                labels.append(label_onehot)
                X_test.append(pic)
            except:
                break
    return np.asarray(X_test).reshape(-1, 256, 256, 1), np.asarray(labels)

def model_test(input_data):
    X_test,_ = test_load_office(input_data)
    model = load_model("model.h5")
    prediction = model.predict(X_test,verbose=1)
    prediction = np.argmax(prediction, axis=1)
    prediction = prediction + 1
    return prediction


model = Sequential()

model.add(Conv2D(16 * 2, (11, 11), activation='relu', input_shape=(256, 256, 1), padding='SAME'))
model.add(Conv2D(16 * 2, (11, 11), activation='relu', padding='SAME'))
# model.add(BatchNormalization())
model.add(
    BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

model.add(Conv2D(32 * 2, (5, 5), activation='relu', padding='SAME'))
model.add(Conv2D(32 * 2, (5, 5), activation='relu', padding='SAME'))
# model.add(BatchNormalization())
model.add(
    BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# 一层卷积层，包含了32个卷积核，大小为3*3
model.add(Conv2D(64 * 2, (3, 3), activation='relu', padding='SAME'))
model.add(Conv2D(64 * 2, (3, 3), activation='relu', padding='SAME'))
# model.add(BatchNormalization())
model.add(
    BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# 添加一个卷积层，包含64个卷积和，每个卷积和仍为3*3
model.add(Conv2D(128 * 2, (3, 3), activation='relu', padding='SAME'))
model.add(Conv2D(128 * 2, (3, 3), activation='relu', padding='SAME'))
# model.add(BatchNormalization())
model.add(
    BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# 添加一个卷积层，包含64个卷积和，每个卷积和仍为3*3
model.add(Conv2D(256 * 2, (3, 3), activation='relu', padding='SAME'))
model.add(Conv2D(256 * 2, (3, 3), activation='relu', padding='SAME'))
model.add(Conv2D(256 * 2, (3, 3), activation='relu', padding='SAME'))
# model.add(BatchNormalization())
model.add(
    BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one'))
model.add(GlobalAveragePooling2D())
# model.add(Dropout(0.25))

# model.add(Dropout(0.25))

# 来一个全连接层
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(5, activation='softmax'))

model.summary()

adam = Adam(lr=1e-4)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

if __name__ == '__main__':

    X_train, Y_train = train_load()
    X_test, Y_test = test_load()
    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)

    save_path = "./model.h5"

    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    datagen.fit(X_train)

    checkpoint = ModelCheckpoint(filepath=save_path, monitor="val_loss", verbose=1, save_best_only=True, mode="auto")
    earlystopping = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')
    callback_list = [checkpoint, earlystopping, TensorBoard("./input/log")]

    model.fit_generator(datagen.flow(X_train, Y_train, batch_size=16),
                        steps_per_epoch=len(X_train) / 16,
                        epochs=1000, verbose=1, callbacks=callback_list,
                        validation_data=(X_test, Y_test))
