import numpy as np

import pandas as pd

from imageio import imread

from skimage.transform import resize

import os



# load image filenames

df_train = pd.read_csv('training.csv')

df_test = pd.read_csv('sample.csv')

train_id = [str(i) for i in df_train['Id']]

test_id = [str(i) for i in df_test['Id']]



width, height = 256, 256



# get and resize training images

train_images = [imread('Images\Training\\' + j) for j in train_id]

resized = [resize(i, (width, height)) for i in train_images]

train_images = np.array(resized)



# get and resize testing images

test_images = [imread('Images\Testing\\' + j) for j in test_id]

resized = [resize(i, (width, height)) for i in test_images]

test_images = np.array(resized)
# augment training data



import random

from skimage import transform



# get Categories for training

y = np.array(df_train['Category'].values)



# generate this many images per training image

augmnt_factor = 10



for i in range(train_images.shape[0]):

    y_array = np.full(augmnt_factor, y[i])

    

#     array_shape = tuple(augmnt_factor) + train_images.shape[1:]

    temp = np.zeros((augmnt_factor, 256, 256, 3))

    

    for j in range(augmnt_factor):

        angle = random.uniform(-30, 30) # get random angle to rotate

        temp[j] = transform.rotate(train_images[i], angle)

    

    y = np.concatenate((y, y_array), axis=0)

    train_images = np.concatenate((train_images, temp), axis=0)
# split training data into train and validate



X_train, X_val = train_images[30:, :, :], train_images[:30, :, :]

y_train, y_val = y[30:], y[:30]
import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D
# define model



model = Sequential()

model.add(Conv2D(8, kernel_size=(11, 11),

                 activation='relu',

                 input_shape=(256,256,3)))

# model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, kernel_size=(11, 11),

                 activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, kernel_size=(5, 5),

                 activation='relu'))

model.add(Conv2D(16, kernel_size=(5, 5),

                 activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(32, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(32, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(1, activation='sigmoid'))
model.compile(loss=keras.losses.binary_crossentropy,

              optimizer=keras.optimizers.Adam(),

              metrics=['accuracy'])
model.fit(X_train, y_train,

          batch_size=200,

          epochs=10,

          verbose=1,

          validation_data=(X_val, y_val))
y_test = np.round(model.predict(test_images))



df_sol = pd.DataFrame()

df_sol['Id'] = df_test['Id']#[str(i[:-4]) for i in df_test['Id']]

df_sol['Category'] = y_test.astype(int)

df_sol.to_csv('sol1.csv', index=False)