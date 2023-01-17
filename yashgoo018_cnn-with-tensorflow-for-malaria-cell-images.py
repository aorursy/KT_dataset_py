!pip install tensorflow-gpu

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import os

from sklearn.model_selection import train_test_split

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Activation, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
base_url = "../input/cell_images/cell_images"
labels = os.listdir(base_url)

label_list = []

data_list = []

for label in range(len(labels)):

  path = os.path.join(base_url, labels[label])

  for image_name in os.listdir(path):

    try:

      image_url = os.path.join(path, image_name)

      image = cv2.imread(image_url)

      image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

      image = cv2.resize(image, (50, 50))

      data_list.append(image)

      label_list.append(label)

    except Exception as e:

      pass

x = np.array(data_list)

y = np.array(label_list)



x_train, x_test, y_train, y_test = train_test_split(x, y)
model = Sequential()



# Convolutional Layer #1

model.add(Conv2D(64, (3, 3), input_shape=(50, 50, 1)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



# Convolutional Layer #2

model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



# Convolutional Layer #3

model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())



model.add(Dense(128))

model.add(Activation('relu'))

model.add(Dropout(0.5))



model.add(Dense(2))

model.add(Activation('softmax'))

          

model.compile(optimizer='adam',

             loss='sparse_categorical_crossentropy',

             metrics=['accuracy'])



model.fit(x_train.reshape(-1, 50, 50, 1), y_train, epochs=3)



model.evaluate(x_test.reshape(-1, 50, 50, 1), y_test)