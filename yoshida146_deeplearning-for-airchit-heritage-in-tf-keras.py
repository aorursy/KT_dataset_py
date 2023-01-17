!pip install -q efficientnet
import pandas as pd

import numpy as np

import os

import joblib
os.listdir('../input/preprocess-for-arichit-heritage/')
train = joblib.load('../input/preprocess-for-arichit-heritage/img_train.pkl')

test = joblib.load('../input/preprocess-for-arichit-heritage/img_test.pkl')

label = joblib.load('../input/preprocess-for-arichit-heritage/label.pkl')



label = np.array(label)
from efficientnet import tfkeras as efn

from tensorflow.keras import Sequential, Model

from tensorflow.keras.layers import Dense, Flatten, Input, Dropout

from tensorflow.keras.applications import ResNet50V2
import tensorflow as tf

from tensorflow.keras.layers import Conv2D, Dropout, Dense, BatchNormalization, Flatten, MaxPooling2D

from tensorflow.keras.models import Sequential
def dnn_model(input_shape=(128, 128, 3), out_shape=10):

    model = Sequential()

    

    # 入力/中間層

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

#     model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))

    model.add(MaxPooling2D(3, 3))

    model.add(Dropout(0.2))

    model.add(BatchNormalization())

    

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

    model.add(MaxPooling2D(3, 3))

    model.add(Dropout(0.2))

    model.add(BatchNormalization())

    

    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))

    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))

    model.add(MaxPooling2D(3, 3))

    model.add(Dropout(0.2))

    model.add(BatchNormalization())

    

    # 出力層

    model.add(Flatten())

    model.add(Dense(units=10, activation='relu'))

    model.add(Dense(units=out_shape, activation='softmax'))

    

    # コンパイル

    model.compile(loss='categorical_crossentropy',

                 optimizer='adam', metrics=['acc'])

    

    return model
def build(shape=(128, 128, 3), n_label=10):

    model = Sequential()

    model.add(Input(shape))

#     model.add(efn.EfficientNetB0(include_top=False, input_shape=shape))

    model.add(ResNet50V2(include_top=False, input_shape=shape))

    

    

#     model.add(Dropout(0.5))

#     model.add(Dense(n_label, activation='softmax'))

    

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['ACC'])

    

    return model
train = np.array(train)

test = np.array(test)
model = dnn_model(train.shape[1: ])
from tensorflow.keras.utils import to_categorical

label_ = to_categorical(label)
train.shape
model.fit(x=train, y=label_, epochs=50)
pred = model.predict_classes(test)
os.listdir('../input/1056lab-archit-heritage-elem-recognit/')
submit = pd.read_csv('../input/1056lab-archit-heritage-elem-recognit/sampleSubmission.csv')

submit['class'] = pred

submit.to_csv('submit.csv', index=False)