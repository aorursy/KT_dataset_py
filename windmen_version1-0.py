from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from tensorflow.python.keras.models import Sequential

from tensorflow.keras.preprocessing import image

from tensorflow.python.keras.layers import Conv2D, MaxPooling2D

from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense

import cv2

from PIL import Image

import matplotlib.pyplot as plt

import imageio

import glob

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_dir = r'/kaggle/input/plagiat/Project/train'

# Каталог с данными для проверки

val_dir =  r'/kaggle/input/plagiat/Project/test'

# Каталог с данными для тестирования

img_width, img_height = 300, 300

# Размерность тензора на основе изображения для входных данных в нейронную сеть

# backend Tensorflow, channels_last

input_shape = (img_width, img_height, 1)

# Количество эпох

epochs = 5

# Размер мини-выборки

batch_size = 2

# Количество изображений для обучения

nb_train_samples = 48

# Количество изображений для проверки

nb_validation_samples = 6
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=input_shape))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())

model.add(Dense(64))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(3))

model.add(Activation('sigmoid'))
model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = datagen.flow_from_directory(

    train_dir,

    target_size=(img_width, img_height),

    batch_size=batch_size,

    class_mode='categorical',

    color_mode="grayscale")
val_generator = datagen.flow_from_directory(

    val_dir,

    target_size=(img_width, img_height),

    batch_size=batch_size,

    class_mode='categorical',

    color_mode="grayscale")
model.fit_generator(

    train_generator,

    steps_per_epoch=nb_train_samples // batch_size,

    epochs=epochs,

    validation_data=val_generator,

    validation_steps=nb_validation_samples // batch_size)
plt.imshow(imageio.imread('/kaggle/input/123456/4%20-%20Copy%201.jpg'))
test_bear = imageio.imread('/kaggle/input/123456/4%20-%20Copy%201.jpg').mean(axis = 2)/255

test_bear = cv2.resize(test_bear, (300,300))

plt.imshow(test_bear)
pred_bear = np.expand_dims(test_bear, axis=2);

pred_bear = np.expand_dims(pred_bear, axis=0);

pred_bear = [pred_bear, pred_bear]

model.predict(pred_bear)
plt.imshow(imageio.imread('/kaggle/input/mon-lis/imgB.jpg'))
test_lisa = imageio.imread('/kaggle/input/mon-lis/imgB.jpg').mean(axis = 2)/255

test_lisa = cv2.resize(test_lisa, (300,300))

plt.imshow(test_lisa)
pred_lisa = np.expand_dims(test_lisa, axis=2);

pred_lisa = np.expand_dims(pred_lisa, axis=0);

pred_lisa = [pred_lisa, pred_lisa]

model.predict(pred_lisa)