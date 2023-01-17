from keras.preprocessing.image import array_to_img, img_to_array, load_img,ImageDataGenerator

import glob

import numpy as np

import cv2
import pandas as pd

import numpy as np

from PIL import Image

import glob

from keras.preprocessing.image import array_to_img, img_to_array, load_img,ImageDataGenerator

from keras.utils import np_utils

from sklearn.model_selection import train_test_split

import keras

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation, Flatten

from keras.layers import BatchNormalization

import matplotlib.pyplot as plt

import matplotlib.cm as cm

from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions

from imblearn.over_sampling import SMOTE

from keras.utils import to_categorical

import tensorflow as tf

from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from google.colab import drive

drive.mount('/content/drive')
%%time

X = []

Y_class = []

Y_def_1 = []

Y_def_2 = []

Y_def_3 = []

Y_def_4 = []

Y_def_5 = []

Y_def_6 = []

X_1 = []

X_2 = []

X_3 = []

X_4 = []

X_5 = []

X_6 = []

img_size = (224,224)

for i in range(7):

    for picture in glob.glob('/content/drive/My Drive/Colab Notebooks/1056lab-defect-detection-extra/class'+str(i)+'/Class'+str(i)+'/*'):

        img = img_to_array(load_img(picture,color_mode = "rgb",  target_size=img_size)) / 255.0

        X.append(img)

        if i == 1:

            Y_class.append(0)

            Y_def_1.append(0)

            X_1.append(img)

        elif i == 2:

            Y_class.append(1)

            Y_def_2.append(0)

            X_2.append(img)

        elif i == 3:

            Y_class.append(2)

            Y_def_3.append(0)

            X_3.append(img)

        elif i == 4:

            Y_class.append(3)

            Y_def_4.append(0)

            X_4.append(img)

        elif i == 5:

            Y_class.append(4)

            Y_def_5.append(0)

            X_5.append(img)

        elif i == 6:

            Y_class.append(5)

            Y_def_6.append(0)

            X_6.append(img)





    for picture in glob.glob('/content/drive/My Drive/Colab Notebooks/1056lab-defect-detection-extra/class'+str(i)+'/Class'+str(i)+'_def_DA/*'):

        img = img_to_array(load_img(picture,color_mode = "rgb",  target_size=img_size)) / 255.0

        X.append(img)

        if i == 1:

            Y_class.append(0)

            Y_def_1.append(1)

            X_1.append(img)

        elif i == 2:

            Y_class.append(1)

            Y_def_2.append(1)

            X_2.append(img)

        elif i == 3:

            Y_class.append(2)

            Y_def_3.append(1)

            X_3.append(img)

        elif i == 4:

            Y_class.append(3)

            Y_def_4.append(1)

            X_4.append(img)

        elif i == 5:

            Y_class.append(4)

            Y_def_5.append(1)

            X_5.append(img)

        elif i == 6:

            Y_class.append(5)

            Y_def_6.append(1)

            X_6.append(img)



# arrayに変換

X = np.array(X)

X_1 = np.array(X_1)

X_2 = np.array(X_2)

X_3 = np.array(X_3)

X_4 = np.array(X_4)

X_5 = np.array(X_5)

X_6 = np.array(X_6)

Y_def_1 = np.array(Y_def_1)

Y_def_2 = np.array(Y_def_2)

Y_def_3 = np.array(Y_def_3)

Y_def_4 = np.array(Y_def_4)

Y_def_5 = np.array(Y_def_5)

Y_def_6 = np.array(Y_def_6)

Y_class = np.array(Y_class)
Y_class = to_categorical(Y_class,6)

Y_def_1 = to_categorical(Y_def_1)

Y_def_2 = to_categorical(Y_def_2)

Y_def_3 = to_categorical(Y_def_3)

Y_def_4 = to_categorical(Y_def_4)

Y_def_5 = to_categorical(Y_def_5)

Y_def_6 = to_categorical(Y_def_6)
conv_layers = VGG16(include_top=False, weights='imagenet', input_shape=(224,224,3))

conv_layers.trainable = False  # 学習させない (学習済みの重みを使う)



# VGG16に全結合層を追加

model = Sequential()

model.add(conv_layers)

model.add(Flatten())

model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(6, activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
epoch = 50

history = model.fit(X, Y_class, epochs=5, batch_size=32,validation_split=0.1)
conv_layers = VGG16(include_top=False, weights='imagenet', input_shape=(224,224,3))

conv_layers.trainable = False  # 学習させない (学習済みの重みを使う)



model_1 = Sequential()

model_1.add(conv_layers)

model_1.add(Flatten())

model_1.add(Dense(1024, activation='relu'))

model_1.add(Dropout(0.5))

model_1.add(Dense(2, activation='sigmoid'))



model_1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history_1 = model_1.fit(X_1, Y_def_1, epochs=epoch, batch_size=32,validation_split=0.3)
conv_layers = VGG16(include_top=False, weights='imagenet', input_shape=(224,224,3))

conv_layers.trainable = False  # 学習させない (学習済みの重みを使う)



# VGG16に全結合層を追加

model_2 = Sequential()

model_2.add(conv_layers)

model_2.add(Flatten())

model_2.add(Dense(1024, activation='relu'))

model_2.add(Dropout(0.5))

model_2.add(Dense(2, activation='sigmoid'))



model_2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history_2 = model_2.fit(X_2, Y_def_2, epochs=epoch, batch_size=64)
conv_layers = VGG16(include_top=False, weights='imagenet', input_shape=(224,224,3))

conv_layers.trainable = False  # 学習させない (学習済みの重みを使う)



# VGG16に全結合層を追加

model_3 = Sequential()

model_3.add(conv_layers)

model_3.add(Flatten())

model_3.add(Dense(1024, activation='relu'))

model_3.add(Dropout(0.5))

model_3.add(Dense(2, activation='sigmoid'))



model_3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history_3 = model_3.fit(X_3, Y_def_3, epochs=epoch, batch_size=32)
conv_layers = VGG16(include_top=False, weights='imagenet', input_shape=(224,224,3))

conv_layers.trainable = False  # 学習させない (学習済みの重みを使う)



# VGG16に全結合層を追加

model_4 = Sequential()

model_4.add(conv_layers)

model_4.add(Flatten())

model_4.add(Dense(1024, activation='relu'))

model_4.add(Dropout(0.5))

model_4.add(Dense(2, activation='sigmoid'))



model_4.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history_4 = model_4.fit(X_4, Y_def_4, epochs=epoch, batch_size=32)
conv_layers = VGG16(include_top=False, weights='imagenet', input_shape=(224,224,3))

conv_layers.trainable = False  # 学習させない (学習済みの重みを使う)



# VGG16に全結合層を追加

model_5 = Sequential()

model_5.add(conv_layers)

model_5.add(Flatten())

model_5.add(Dense(1024, activation='relu'))

model_5.add(Dropout(0.5))

model_5.add(Dense(2, activation='sigmoid'))



model_5.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history_5 = model_5.fit(X_5, Y_def_5, epochs=epoch, batch_size=32)
conv_layers = VGG16(include_top=False, weights='imagenet', input_shape=(224,224,3))

conv_layers.trainable = False  # 学習させない (学習済みの重みを使う)



# VGG16に全結合層を追加

model_6 = Sequential()

model_6.add(conv_layers)

model_6.add(Flatten())

model_6.add(Dense(1024, activation='relu'))

model_6.add(Dropout(0.5))

model_6.add(Dense(2, activation='sigmoid'))



model_6.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history_6 = model_6.fit(X_6, Y_def_6, epochs=epoch, batch_size=32)
p = []

test_img = glob.glob('/content/drive/My Drive/Colab Notebooks/1056lab-defect-detection-extra/test/*')

test_img.sort()

# 対象Aの画像

for picture in test_img:

    img = img_to_array(load_img(picture,color_mode = 'rgb', target_size=img_size)) / 255.0

    test_data = np.array(img)

    test_data = np.reshape(test_data,[1,224,224,3])

    c = model.predict_classes(test_data)

    if c == 0:

        p.append(model_1.predict(test_data)[:,1])

    elif c == 1:

        p.append(model_2.predict_proba(test_data)[:,1])

    elif c == 2:

        p.append(model_3.predict_proba(test_data)[:,1])

    elif c == 3:

        p.append(model_4.predict_proba(test_data)[:,1])

    elif c == 4:

        p.append(model_5.predict_proba(test_data)[:,1])

    elif c == 5:

        p.append(model_6.predict_proba(test_data)[:,1])
p = np.array(p)
p = p.reshape(2070)
p = p.tolist()
submit = pd.read_csv('/content/drive/My Drive/Colab Notebooks/1056lab-defect-detection-extra/sampleSubmission.csv')

submit['defect'] = p
submit = submit.set_index('name')
with open('/content/drive/My Drive/Colab Notebooks/1056lab-defect-detection-extra/submit/submission2.csv', 'w', encoding = 'utf-8-sig') as f:

  submit.to_csv(f)

!cat /content/drive/My Drive/Colab Notebooks/1056lab-defect-detection-extra/submit/submission2.csv