import os

import math



import numpy as np

import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow.keras.applications import DenseNet121

from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import layers, optimizers

from sklearn.model_selection import train_test_split

import cv2
load_dir = '/kaggle/input/bengaliai/256_train/256/'



train = pd.read_csv('/kaggle/input/bengaliai-cv19/train.csv')

train['filename'] = train.image_id.apply(lambda filename: load_dir + filename + '.png')



train.head()
def get_pad_width(im, new_shape, is_rgb=True):

    pad_diff = new_shape - im.shape[0], new_shape - im.shape[1]

    t, b = math.floor(pad_diff[0]/2), math.ceil(pad_diff[0]/2)

    l, r = math.floor(pad_diff[1]/2), math.ceil(pad_diff[1]/2)

    if is_rgb:

        pad_width = ((t,b), (l,r), (0, 0))

    else:

        pad_width = ((t,b), (l,r))

    return pad_width
def crop_object(img, thresh=220, maxval=255, square=True):

    """

    Source: https://stackoverflow.com/questions/49577973/how-to-crop-the-biggest-object-in-image-with-python-opencv

    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale

    # threshold to get just the signature (INVERTED)

    retval, thresh_gray = cv2.threshold(gray, thresh=thresh, maxval=maxval, type=cv2.THRESH_BINARY_INV)



    contours, hierarchy = cv2.findContours(thresh_gray,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)



    # Find object with the biggest bounding box

    mx = (0,0,0,0)      # biggest bounding box so far

    mx_area = 0

    for cont in contours:

        x,y,w,h = cv2.boundingRect(cont)

        area = w*h

        if area > mx_area:

            mx = x,y,w,h

            mx_area = area

    x,y,w,h = mx

    

    crop = img[y:y+h, x:x+w]

    

    if square:

        pad_width = get_pad_width(crop, max(crop.shape))

        crop = np.pad(crop, pad_width=pad_width, mode='constant', constant_values=255)

    

    return crop
def data_generator(filenames, y, batch_size=64, shape=(128, 128, 1), random_state=2019):

    y = y.copy()

    np.random.seed(random_state)

    indices = np.arange(len(filenames))

    

    while True:

        np.random.shuffle(indices)

        

        for i in range(0, len(indices), batch_size):

            batch_idx = indices[i:i+batch_size]

            size = len(batch_idx)

            

            batch_files = filenames[batch_idx]

            X_batch = np.zeros((size, *shape))

            y_batch = y[batch_idx]

            

            for i, file in enumerate(batch_files):

                img = cv2.imread(file)

                img = crop_object(img, thresh=220)

                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                img = cv2.resize(img, shape[:2])

                X_batch[i, :, :, 0] = img / 255.

            

            yield X_batch, [y_batch[:, i] for i in range(y_batch.shape[1])]
def build_model(densenet):

    x_in = layers.Input(shape=(128, 128, 1))

    x = layers.Conv2D(3, (3, 3), padding='same')(x_in)

    x = densenet(x)

    

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dropout(0.5)(x)

    x = layers.Dense(256, activation='relu')(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dropout(0.5)(x)

    

    out_grapheme = layers.Dense(168, activation='softmax', name='grapheme')(x)

    out_vowel = layers.Dense(11, activation='softmax', name='vowel')(x)

    out_consonant = layers.Dense(7, activation='softmax', name='consonant')(x)

    

    model = Model(inputs=x_in, outputs=[out_grapheme, out_vowel, out_consonant])

    

    model.compile(

        optimizers.Adam(lr=0.0001), 

        metrics=['accuracy'], 

        loss='sparse_categorical_crossentropy'

    )

    

    return model
weights_path = '/kaggle/input/densenet-keras/DenseNet-BC-121-32-no-top.h5'

densenet = DenseNet121(include_top=False, weights=weights_path, input_shape=(128, 128, 3))
model = build_model(densenet)

model.summary()
train_files, valid_files, y_train, y_valid = train_test_split(

    train.filename.values, 

    train[['grapheme_root','vowel_diacritic', 'consonant_diacritic']].values, 

    test_size=0.25, 

    random_state=2019

)
batch_size = 128



train_gen = data_generator(train_files, y_train)

valid_gen = data_generator(valid_files, y_valid)



train_steps = round(len(train_files) / batch_size) + 1

valid_steps = round(len(valid_files) / batch_size) + 1
callbacks = [tf.keras.callbacks.ModelCheckpoint('model.h5', save_best_only=True)]



train_history = model.fit_generator(

    train_gen,

    steps_per_epoch=train_steps,

    epochs=15,

    validation_data=valid_gen,

    validation_steps=valid_steps,

    callbacks=callbacks

)
pd.DataFrame(train_history.history).to_csv('history.csv', index=False)