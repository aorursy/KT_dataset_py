# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.image as img

from keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Flatten, Dense, Dropout,Input, ZeroPadding2D

from keras.models import Sequential

from keras.layers import Dense, Activation

from keras.utils import to_categorical

from keras.utils import plot_model

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import seaborn as sns

from keras.preprocessing.image import load_img, img_to_array

import matplotlib.pyplot as plt

import os



plt.figure(0, figsize=(12,20))

cpt = 0



cpt = cpt + 1

plt.subplot(7,5,cpt)

img = load_img("/kaggle/input/horses-or-humans-dataset/horse-or-human/train/horses/horse38-2.png")

plt.imshow(img, cmap="gray")



cpt = cpt + 2

plt.subplot(7,5,cpt)

img1 = load_img("/kaggle/input/horses-or-humans-dataset/horse-or-human/train/humans/human16-26.png")

plt.imshow(img1, cmap="gray")



plt.tight_layout()

plt.show()
from keras.preprocessing.image import ImageDataGenerator

batch_size = 128



datagen = ImageDataGenerator(rescale=1./255)

train_it = datagen.flow_from_directory('/kaggle/input/horses-or-humans-dataset/horse-or-human/horse-or-human/train/', subset='training',class_mode='binary')

val_it = datagen.flow_from_directory('/kaggle/input/horses-or-humans-dataset/horse-or-human/horse-or-human/validation/', class_mode='binary')

batchX, batchy = train_it.next()

print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))
#create a model

def buildModel(): 

    model = Sequential()

    

    model.add(Conv2D(32, (3, 3), input_shape=(256, 256, 3)))

    model.add(Activation('relu'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(2))

    

    model.add(Conv2D(32, (3, 3)))

    model.add(Activation('relu'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(2))



    model.add(Conv2D(64, (3, 3)))

    model.add(Activation('relu'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(2))

    

    model.add(Conv2D(128, (5, 5)))

    model.add(Activation('relu'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(2))

    

    model.add(Flatten())

    model.add(Dense(64))

    model.add(Activation('relu'))

    model.add(Dropout(0.5))

    model.add(Dense(1))

    model.add(Activation('sigmoid'))

    

    return model;      
model = buildModel()

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
model.summary()
#With validation data, compare against validation data (dev - test)

#If steps_per_epoch is set, the `batch_size` must be None.

#batch size = 2**x , 16,32,64,24 

epoch = 25

steps_per_epoch = 25

learning_rate = 0.01

validation_steps = 15

batch_size = 32
from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint("model_hh.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

callbacks_list = [checkpoint]
from keras.optimizers import SGD

model_sgd = buildModel()

optimizer = SGD(lr=learning_rate, nesterov=True)

model_sgd.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=["accuracy"])

history = model_sgd.fit_generator(train_it,epochs=epoch, steps_per_epoch = steps_per_epoch,validation_data=val_it, validation_steps=validation_steps,callbacks=callbacks_list)
import matplotlib.pyplot as plt



acc = history.history['accuracy']

loss = history.history['loss']



val_acc = history.history['val_accuracy']

val_loss = history.history['val_loss']



epochs = range(1, len(acc) + 1)



plt.figure(figsize = (16, 5))



plt.subplot(1,2,1)

plt.plot(epochs, acc, 'r', label = 'Training Accuracy')

plt.plot(epochs, val_acc, 'b', label = 'Validation Accuracy')

plt.title('Training vs. Validation Accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend()



plt.subplot(1,2,2)

plt.plot(epochs, loss, 'r', label = 'Training Loss')

plt.plot(epochs, val_loss, 'b', label = 'Validation Loss')

plt.title('Training vs. Validation Loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend()



plt.show()
from keras.callbacks import ModelCheckpoint

checkpoint_prop = ModelCheckpoint("model_prp.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

callbacks_list_prop = [checkpoint_prop]
from keras.optimizers import RMSprop

model_rmsprop = buildModel()

optimizer = RMSprop(lr=learning_rate)

model_rmsprop.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=["accuracy"])

history = model_rmsprop.fit_generator(train_it,epochs=epoch, steps_per_epoch = steps_per_epoch,validation_data=val_it, validation_steps=validation_steps,callbacks=callbacks_list_prop)
acc = history.history['accuracy']

loss = history.history['loss']



val_acc = history.history['val_accuracy']

val_loss = history.history['val_loss']



epochs = range(1, len(acc) + 1)



plt.figure(figsize = (16, 5))



plt.subplot(1,2,1)

plt.plot(epochs, acc, 'r', label = 'Training Accuracy')

plt.plot(epochs, val_acc, 'b', label = 'Validation Accuracy')

plt.title('Training vs. Validation Accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend()



plt.subplot(1,2,2)

plt.plot(epochs, loss, 'r', label = 'Training Loss')

plt.plot(epochs, val_loss, 'b', label = 'Validation Loss')

plt.title('Training vs. Validation Loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend()



plt.show()
from keras.callbacks import ModelCheckpoint

checkpoint_adam = ModelCheckpoint("model_ad.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

callbacks_list_adm = [checkpoint_adam]
from keras.optimizers import Adam

model_adam = buildModel()

optimizer = Adam(lr=learning_rate)

model_adam.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=["accuracy"])

history = model_adam.fit_generator(train_it,epochs=epoch, steps_per_epoch = steps_per_epoch,validation_data=val_it, validation_steps=validation_steps,callbacks=callbacks_list_adm)
acc = history.history['accuracy']

loss = history.history['loss']



val_acc = history.history['val_accuracy']

val_loss = history.history['val_loss']



epochs = range(1, len(acc) + 1)



plt.figure(figsize = (16, 5))



plt.subplot(1,2,1)

plt.plot(epochs, acc, 'r', label = 'Training Accuracy')

plt.plot(epochs, val_acc, 'b', label = 'Validation Accuracy')

plt.title('Training vs. Validation Accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend()



plt.subplot(1,2,2)

plt.plot(epochs, loss, 'r', label = 'Training Loss')

plt.plot(epochs, val_loss, 'b', label = 'Validation Loss')

plt.title('Training vs. Validation Loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend()



plt.show()
