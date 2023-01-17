# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import cv2

import numpy as np

from keras import models

from keras import layers

from keras import optimizers

from keras.applications import VGG16

from keras.callbacks import ReduceLROnPlateau

from keras.preprocessing.image import ImageDataGenerator



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
def preprocess_input(x):

    x = x.astype('float32')

    # 我是用cv2来读取的图片，其已经是BGR格式了

    mean = [103.939, 116.779, 123.68]

    x -= mean

    return x





def load_data():

    # 这是统计学专家提供的训练集

    data = np.load('../input/captcha.npz')

    train_x, train_y = data['images'], data['labels']

    train_x = preprocess_input(train_x)

    # 由于是统计得来的信息，所以在此给定可信度

    sample_weight = train_y.max(axis=1) / np.sqrt(train_y.sum(axis=1))

    sample_weight /= sample_weight.mean()

    train_y = train_y.argmax(axis=1)



    # 这是人工提供的验证集

    data = np.load('../input/captcha.test.npz')

    test_x, test_y = data['images'], data['labels']

    # resize

    n = test_x.shape[0]

    new_test_x = np.empty((n, 67, 67, 3), dtype=np.uint8)

    for idx in range(n):

        new_test_x[idx] = cv2.resize(test_x[idx], (67, 67))

    test_x = preprocess_input(new_test_x)

    return (train_x, train_y, sample_weight), (test_x, test_y)
def learn():

    (train_x, train_y, sample_weight), (test_x, test_y) = load_data()

    datagen = ImageDataGenerator(horizontal_flip=True,

                                 vertical_flip=True)

    train_generator = datagen.flow(train_x, train_y, sample_weight=sample_weight)

    _, h, w, c = train_x.shape

    base = VGG16(weights='imagenet', include_top=False, input_shape=(h, w, c))

    for layer in base.layers[:-4]:

        layer.trainable = False

    model = models.Sequential([

        base,

        layers.BatchNormalization(),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),

        layers.GlobalAveragePooling2D(),

        layers.BatchNormalization(),

        layers.Dense(64, activation='relu'),

        layers.BatchNormalization(),

        layers.Dropout(0.20),

        layers.Dense(80, activation='softmax')

    ])

    model.compile(optimizer=optimizers.RMSprop(lr=1e-5),

                  loss='sparse_categorical_crossentropy',

                  metrics=['accuracy'])

    model.summary()

    reduce_lr = ReduceLROnPlateau(verbose=1)

    model.fit_generator(train_generator, epochs=400,

                        steps_per_epoch=100,

                        validation_data=(test_x[:800], test_y[:800]),

                        callbacks=[reduce_lr],

                        verbose=0)

    result = model.evaluate(test_x, test_y)

    print(result)
learn()