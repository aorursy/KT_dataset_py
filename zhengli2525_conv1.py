# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from __future__ import print_function

import keras

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D

import os

import pickle

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from keras.callbacks import ModelCheckpoint





X=pd.read_pickle("/kaggle/input/zoom-32-32-1/datas.pickle")

y=pd.read_pickle("/kaggle/input/zoom-32-32-1/labels.pickle")



le=preprocessing.LabelEncoder()



y_labels=le.fit_transform(y)







X_train, X_test, y_train, y_test = train_test_split(X, y_labels, test_size=.5, random_state=42)



model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',

                 input_shape=X_train.shape[1:],name='1_conv'))

model.add(Activation('relu',name='1_activation'))

model.add(Conv2D(32, (3, 3),name='2_conv'))

model.add(Activation('relu',name='2_activation'))

model.add(MaxPooling2D(pool_size=(2, 2),name='2_pooling'))

model.add(Dropout(0.25,name='2_dropout'))



model.add(Conv2D(64, (3, 3), padding='same',name='3_conv'))

model.add(Activation('relu',name='3_activation'))

model.add(Conv2D(64, (3, 3),name='4_conf'))

model.add(Activation('relu',name='4_activation'))

model.add(MaxPooling2D(pool_size=(2, 2),name='4_pooling'))

model.add(Dropout(0.25,name='4_dropout'))



model.add(Flatten())

model.add(Dense(1152,name='5_dense'))

model.add(Activation('relu',name='5_activation'))

model.add(Dropout(0.5,name='5_dropout'))

model.add(Dense(1118,name='6_dense'))

model.add(Activation('softmax',name='6_activation'))



# initiate RMSprop optimizer

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)



# Let's train the model using RMSprop

model.compile(loss='sparse_categorical_crossentropy',

              optimizer=opt,

              metrics=['accuracy'])



model.summary()
datagen = ImageDataGenerator(

        featurewise_center=True,  # set input mean to 0 over the dataset

        samplewise_center=True,  # set each sample mean to 0

        featurewise_std_normalization=True,  # divide inputs by std of the dataset

        samplewise_std_normalization=True,  # divide each input by its std

        width_shift_range=0.1,

        height_shift_range=0.1,

        rescale=1./255,

        fill_mode='nearest')



datagen.fit(X_train)



testgen=ImageDataGenerator=ImageDataGenerator(

        rescale=1./255)



testgen.fit(X_test)



callbacks_list = [

    keras.callbacks.EarlyStopping(

    monitor='acc',

    patience=1,

    ),

    keras.callbacks.ModelCheckpoint(

    filepath='my_model.h5',

    monitor='val_loss',

    save_best_only=True,

    )]





history=model.fit_generator(datagen.flow(X_train, y_train,

                        batch_size=559),

                        steps_per_epoch=77,

                        epochs=100,

                        validation_data=testgen.flow(X_test,y_test,batch_size=559),

                        validation_steps=77,

                        callbacks=callbacks_list,

                        workers=4)





scores = model.evaluate(X_test, y_test, verbose=1)

print('Test loss:', scores[0])

print('Test accuracy:', scores[1])
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')



acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))





plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

axes=plt.gca()

axes.set_xlim([0,len(epochs)])

axes.set_ylim([0,max(acc)])

plt.title('Training and validation accuracy')

plt.legend()

plt.axes()

plt.figure()



plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

axes=plt.gca()

axes.set_xlim([0,len(epochs)])

axes.set_ylim([min(loss),max(loss)])

plt.legend()

plt.axes()

plt.show()