import os

import sys

# Repository source: https://github.com/qubvel/efficientnet

sys.path.append(os.path.abspath('../input/efficientnet/efficientnet-master/efficientnet-master/'))
from efficientnet import EfficientNetB3

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten 

from keras.models import Model

from keras import optimizers

from keras.utils import np_utils

from keras.preprocessing.image import ImageDataGenerator

import cv2

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from pathlib import Path

%matplotlib inline
%ls ../input/Kannada-MNIST/
root_path = Path("../input/Kannada-MNIST/")
# Load the data

train = pd.read_csv(root_path / "train.csv")

test = pd.read_csv(root_path / "test.csv")
train.head()
Y_train = train["label"]



# Drop 'label' column

X_train = train.drop(labels = ["label"],axis = 1) 



test.drop('id', axis=1, inplace=True)
# Normilize data

X_train = X_train.astype('float32')

test = test.astype('float32')

X_train /= 255

test /= 255
X_train = X_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
X_train3 = np.full((X_train.shape[0], 28, 28, 3), 0.0)



for i, s in enumerate(X_train):

    X_train3[i] = cv2.cvtColor(s, cv2.COLOR_GRAY2RGB) 
test3 = np.full((test.shape[0], 28, 28, 3), 0.0)



for i, s in enumerate(test):

    test3[i] = cv2.cvtColor(s, cv2.COLOR_GRAY2RGB) 
X_train3.shape, test3.shape
Y_train = np_utils.to_categorical(Y_train, 10)
%ls ../input/efficientnet-keras-weights-b0b5/
# Load in EfficientNetB3

model = EfficientNetB3(weights=None,

                        include_top=False,

                        input_shape=(28, 28, 3))

model.load_weights('../input/efficientnet-keras-weights-b0b5/efficientnet-b3_imagenet_1000_notop.h5')
model.trainable = False
x = model.output

x = Flatten()(x)

x = Dense(1024, activation="relu")(x)

x = Dropout(0.5)(x)

predictions = Dense(units = 10, activation="softmax")(x)

model_f = Model(input = model.input, output = predictions)

model_f.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),

                loss='categorical_crossentropy',

                metrics=['accuracy'])
datagen = ImageDataGenerator(

        rotation_range= 8,  

        zoom_range = 0.15,  

        width_shift_range=0.2, 

        height_shift_range=0.2)

datagen.fit(X_train3)
from sklearn.model_selection import train_test_split

X_train, X_val1, y_train, y_val1 = train_test_split(

    X_train3, Y_train, test_size=0.05, random_state=42)
from keras.callbacks import ModelCheckpoint

size_batch = 60

checkpoint = ModelCheckpoint('BWeight.md5',monitor='val_loss',

                            save_best_only=True)
history = model_f.fit_generator(datagen.flow(X_train,y_train, batch_size=size_batch),

                              epochs = 50,

                              validation_data = (X_val1,y_val1),

                              verbose = 2,

                              steps_per_epoch = X_train.shape[0] // size_batch,

                              callbacks=[checkpoint])
import json



with open('history.json', 'w') as f:

    json.dump(history.history, f)



history_df = pd.DataFrame(history.history)

history_df[['loss', 'val_loss']].plot()

history_df[['acc', 'val_acc']].plot()
%%time

# Prediction

test_predictions = model_f.predict(test3)
# select the index with the maximum probability

results = np.argmax(test_predictions,axis = 1)
submission = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')

submission['label'] = results

submission.to_csv('submission.csv', index=False)

submission.head()