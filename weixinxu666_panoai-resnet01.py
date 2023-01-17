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
import numpy as np
from keras import layers
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, \
    AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.utils import plot_model
from keras.initializers import glorot_uniform
from keras.optimizers import Adam
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from keras_preprocessing.image import ImageDataGenerator
from matplotlib.pyplot import imshow

import keras.backend as K

K.set_image_data_format('channels_last')

file = '../input/panoai2'
shuffle = 10000


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

def identity_block(X, f, filters, stage, block):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def convolutional_block(X, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    X_shortcut = Conv2D(F3, (1, 1), strides=(s, s), name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


tf.reset_default_graph()

with tf.Session() as sess:
    np.random.seed(1)
    A_prev = tf.placeholder("float", [3, 4, 4, 6])
    X = np.random.randn(3, 4, 4, 6)
    A = identity_block(A_prev, f=2, filters=[2, 4, 6], stage=1, block='a')
    sess.run(tf.global_variables_initializer())
    out = sess.run([A], feed_dict={A_prev: X, K.learning_phase(): 0})
    print("out = " + str(out[0][1][1][0]))


def ResNet50(input_shape=(256, 256, 1), classes=5):
    X_input = Input(input_shape)

    X = ZeroPadding2D((3, 3))(X_input)

    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    X = AveragePooling2D((2, 2), name='avg_pool')(X)

    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs=X_input, outputs=X, name='ResNet50')

    return model

model = ResNet50(input_shape = (256, 256, 1), classes = 5)

model.summary()

adam = Adam(lr=1e-4)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

if __name__ == '__main__':

    X_train, Y_train = train_load()
    X_test, Y_test = test_load()
    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)

    save_path = "model.h5"

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
    callback_list = [checkpoint, earlystopping, TensorBoard("")]

    model.fit_generator(datagen.flow(X_train, Y_train, batch_size=16),
                        steps_per_epoch=len(X_train) / 16,
                        epochs=1000, verbose=1, callbacks=callback_list,
                        validation_data=(X_test, Y_test))
