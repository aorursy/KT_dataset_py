# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import keras
from keras import layers
from keras import models
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from os import listdir, makedirs
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras import optimizers, regularizers
from keras import losses
from keras.preprocessing import image
from keras.datasets import mnist

import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv("../input/the-simpsons-characters-dataset/number_pic_char.csv")
data.head(10)
RESOLUTION = 64
BATCH_SIZE = 16

#if you need data augmentation processing
#train_datagen = ImageDataGenerator(
        #rescale=1./255,
        #shear_range=0.2,
        #zoom_range=0.2,
        #horizontal_flip=True,
        #validation_split=0.3)

data_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.15)

train_generator = data_datagen.flow_from_directory(
        "../input/the-simpsons-characters-dataset/simpsons_dataset/simpsons_dataset",
        classes=['homer_simpson', 'ned_flanders', 'moe_szyslak', 'lisa_simpson', 
                 'bart_simpson', 'marge_simpson', 'krusty_the_clown', 
                 'principal_skinner', 'charles_montgomery_burns', 'milhouse_van_houten'],
        target_size=(RESOLUTION, RESOLUTION),
        batch_size=BATCH_SIZE,
        class_mode='categorical', subset="training")

val_generator = data_datagen.flow_from_directory(
        "../input/the-simpsons-characters-dataset/simpsons_dataset/simpsons_dataset",
        classes=['homer_simpson', 'ned_flanders', 'moe_szyslak', 'lisa_simpson', 
                 'bart_simpson', 'marge_simpson', 'krusty_the_clown', 
                 'principal_skinner', 'charles_montgomery_burns', 'milhouse_van_houten'],
        target_size=(RESOLUTION, RESOLUTION),
        batch_size=BATCH_SIZE,
        class_mode='categorical', subset="validation")
model = models.Sequential()
model.add(layers.Conv2D(filters= 100, kernel_size=(5, 5), activation='relu', input_shape=(64, 64, 3))) #(image_height, image_width, image_channels) (not including the batch dimension).
model.add(layers.Conv2D(filters= 100, kernel_size=(5, 5), activation='relu'))
model.add(layers.MaxPooling2D((4, 4)))
model.add(layers.Flatten()) # Output_shape=(None, 3*3*64)
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='relu'))

model.summary()
model1 = model
model1.compile(loss='categorical_crossentropy', optimizer=optimizers.adagrad(lr=0.001), metrics=['acc'])

history = model1.fit_generator(
        train_generator,
        steps_per_epoch=(11745 // 128),
        epochs=10,
        validation_data=val_generator,
        validation_steps=(2066 // 128) 
    )
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
model2 = model
model2.compile(loss='categorical_crossentropy', optimizer=optimizers.adagrad(lr=0.01), metrics=['acc'])

history2 = model2.fit_generator(
        train_generator,
        steps_per_epoch=(11745 // 128),
        epochs=10,
        validation_data=val_generator,
        validation_steps=(2066 // 128) 
    )
acc = history2.history['acc']
val_acc = history2.history['val_acc']
loss = history2.history['loss']
val_loss = history2.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
model3 = model
model3.compile(loss='categorical_crossentropy', optimizer=optimizers.adagrad(lr=0.1), metrics=['acc'])

history3 = model3.fit_generator(
        train_generator,
        steps_per_epoch=(11745 // 128),
        epochs=10,
        validation_data=val_generator,
        validation_steps=(2066 // 128) 
    )
acc = history3.history['acc']
val_acc = history3.history['val_acc']
loss = history3.history['loss']
val_loss = history3.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
model4 = model
model4.compile(loss='categorical_crossentropy', optimizer=optimizers.adagrad(lr=0.001), metrics=['acc'])

history4 = model4.fit_generator(
        train_generator,
        steps_per_epoch=(11745 // 128),
        epochs=50,
        validation_data=val_generator,
        validation_steps=(2066 // 128) 
    )
acc = history4.history['acc']
val_acc = history4.history['val_acc']
loss = history4.history['loss']
val_loss = history4.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
model5 = model
model5.compile(loss='categorical_crossentropy', optimizer=optimizers.adagrad(lr=0.01), metrics=['acc'])

history5 = model5.fit_generator(
        train_generator,
        steps_per_epoch=(11745 // 128),
        epochs=50,
        validation_data=val_generator,
        validation_steps=(2066 // 128) 
    )
acc = history5.history['acc']
val_acc = history5.history['val_acc']
loss = history5.history['loss']
val_loss = history5.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
model6 = model
model6.compile(loss='categorical_crossentropy', optimizer=optimizers.adagrad(lr=0.1), metrics=['acc'])

history6 = model6.fit_generator(
        train_generator,
        steps_per_epoch=(11745 // 128),
        epochs=50,
        validation_data=val_generator,
        validation_steps=(2066 // 128) 
    )
acc = history6.history['acc']
val_acc = history6.history['val_acc']
loss = history6.history['loss']
val_loss = history6.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
model7 = model
model7.compile(loss='categorical_crossentropy', optimizer=optimizers.adagrad(lr=0.001), metrics=['acc'])

history7 = model7.fit_generator(
        train_generator,
        steps_per_epoch=(11745 // 128),
        epochs=100,
        validation_data=val_generator,
        validation_steps=(2066 // 128) 
    )
acc = history7.history['acc']
val_acc = history7.history['val_acc']
loss = history7.history['loss']
val_loss = history7.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
model8 = model
model8.compile(loss='categorical_crossentropy', optimizer=optimizers.adagrad(lr=0.01), metrics=['acc'])

history8 = model8.fit_generator(
        train_generator,
        steps_per_epoch=(11745 // 128),
        epochs=10,
        validation_data=val_generator,
        validation_steps=(2066 // 128) 
    )
acc = history8.history['acc']
val_acc = history8.history['val_acc']
loss = history8.history['loss']
val_loss = history8.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
model9 = model
model9.compile(loss='categorical_crossentropy', optimizer=optimizers.adagrad(lr=0.1), metrics=['acc'])

history9 = model9.fit_generator(
        train_generator,
        steps_per_epoch=(11745 // 128),
        epochs=100,
        validation_data=val_generator,
        validation_steps=(2066 // 128) 
    )
acc = history9.history['acc']
val_acc = history9.history['val_acc']
loss = history9.history['loss']
val_loss = history9.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
