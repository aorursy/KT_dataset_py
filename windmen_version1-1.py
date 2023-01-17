from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from tensorflow.python.keras.models import Sequential

from tensorflow.keras.preprocessing import image

from tensorflow.python.keras.layers import Conv2D, MaxPooling2D

from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense

from PIL import Image

import cv2

import shutil

import matplotlib.pyplot as plt

import imageio

import glob

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Каталог с набором данных

data_dir = '/kaggle/input/test-data/Test_dataset'

# Каталог с данными для обучения

train_dir = 'train'

# Каталог с данными для проверки

val_dir = 'val'

# Каталог с данными для тестирования

test_dir = 'test'

# Часть набора данных для тестирования

test_data_portion = 0.15

# Часть набора данных для проверки

val_data_portion = 0.15

# Количество элементов данных в одном классе

nb_images = 153

nb_test_samples = 69
def create_directory(dir_name):

    if os.path.exists(dir_name):

        shutil.rmtree(dir_name)

    os.makedirs(dir_name)

    os.makedirs(os.path.join(dir_name, "bears"))

    os.makedirs(os.path.join(dir_name, "lisas"))

    os.makedirs(os.path.join(dir_name, "creations"))
create_directory(train_dir)

create_directory(val_dir)

create_directory(test_dir)
def copy_images(start_index, end_index, source_dir, dest_dir):

    for i in range(start_index, end_index):

        shutil.copy2(os.path.join(source_dir+'/bearsInTheForest/', "pic_" + str(i) + ".jpg"), 

                    os.path.join(dest_dir, "bears"))

        shutil.copy2(os.path.join(source_dir+'/monaLisa/', "pic_" + str(i) + ".jpg"), 

                   os.path.join(dest_dir, "lisas"))

        shutil.copy2(os.path.join(source_dir+'/theCreationOfAdam/', "pic_" + str(i) + ".jpg"), 

                   os.path.join(dest_dir, "creations"))
start_val_data_idx = int(nb_images * (1 - val_data_portion - test_data_portion))

start_test_data_idx = int(nb_images * (1 - test_data_portion))

print(start_val_data_idx)

print(start_test_data_idx)
copy_images(0, start_val_data_idx, data_dir, train_dir)

copy_images(start_val_data_idx, start_test_data_idx, data_dir, val_dir)

copy_images(start_test_data_idx, nb_images, data_dir, test_dir)
# Каталог с данными для обучения

train_dir = r'/kaggle/working/train'

# Каталог с данными для проверки

val_dir =  r'/kaggle/working/val'

# Каталог с данными для тестирования

test_dir = r'/kaggle/working/test'

img_width, img_height = 300, 300

# Размерность тензора на основе изображения для входных данных в нейронную сеть

# backend Tensorflow, channels_last

input_shape = (img_width, img_height, 1)

# Количество эпох

epochs = 10

# Размер мини-выборки

batch_size = 2

# Количество изображений для обучения

nb_train_samples = 321

# Количество изображений для проверки

nb_validation_samples = 69
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=input_shape))

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
test_generator = datagen.flow_from_directory(

    test_dir,

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
def create_directory(dir_name):

    if os.path.exists(dir_name):

        shutil.rmtree(dir_name)

    os.makedirs(dir_name)

    os.makedirs(os.path.join(dir_name, "cats"))

    os.makedirs(os.path.join(dir_name, "dogs"))
train_dir = '/kaggle/input/test1'
scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)

print("Аккуратность на тестовых данных: %.2f%%" % (scores[1]*100))