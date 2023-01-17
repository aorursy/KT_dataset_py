# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import warnings

warnings.filterwarnings('ignore')



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import keras

from keras import layers

from keras.layers.normalization import BatchNormalization

from keras.models import Model

from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

import os



from sklearn.model_selection import train_test_split



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

sub = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")
train.shape
X_train = train.drop(columns=['label']).values.reshape(-1, 28, 28, 1)

y_train = train.label.values

X_test = test.values.reshape(-1, 28, 28, 1)
X_train = X_train / 255.0

X_test = X_test / 255.0
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.1, random_state=39)
datagen = ImageDataGenerator(rotation_range=20,zoom_range = 0.1, width_shift_range=0.1, height_shift_range=0.1)

datagen.fit(X_train)
callbacks_list = [

  keras.callbacks.ReduceLROnPlateau(

  # This callback will monitor the validation loss of the model

  monitor='val_loss',

  # It will divide the learning by 10 when it gets triggered

  factor=0.1,

  # It will get triggered after the validation loss has stopped improving

  # for at least 10 epochs

  patience=5,

  ),

  keras.callbacks.EarlyStopping(

      monitor='val_loss', 

      min_delta=1e-3, 

      patience=15, 

      verbose=0, 

      mode='auto', 

      baseline=None, 

      restore_best_weights=False)

]
def IdentityBlock(x, filters=32, kernel_size=3):

    x_shortcut = x

    x = layers.Conv2D(filters=filters, kernel_size=(kernel_size,kernel_size), padding="Same")(x)

    x = layers.BatchNormalization()(x)

    x = layers.Activation('relu')(x)



    x = layers.Conv2D(filters=filters, kernel_size=(kernel_size,kernel_size), padding="Same")(x)

    x = layers.BatchNormalization()(x)

    x = layers.Activation('relu')(x)

    

    x = layers.Conv2D(filters=filters, kernel_size=(kernel_size,kernel_size), padding="Same")(x)

    x = layers.BatchNormalization()(x)

    x = layers.Add()([x, x_shortcut])

    x = layers.Activation('relu')(x)

    x = layers.Dropout(0.25)(x)

    

    return x



def ConvBlock(x, filters=32, kernel_size=3):

    x_shortcut = x

    x = layers.Conv2D(filters=filters, kernel_size=(kernel_size,kernel_size), padding="Same")(x)

    x = layers.BatchNormalization()(x)

    x = layers.Activation('relu')(x)

    

    x = layers.Conv2D(filters=filters, kernel_size=(kernel_size,kernel_size), padding="Same")(x)

    x = layers.BatchNormalization()(x)

    x = layers.Activation('relu')(x)

    

    x = layers.Conv2D(filters=filters, kernel_size=(kernel_size,kernel_size), padding="Same")(x)

    x = layers.BatchNormalization()(x)

    

    x_shortcut = layers.Conv2D(filters=filters, kernel_size=(kernel_size,kernel_size), padding="Same")(x_shortcut)

    x_shortcut = layers.BatchNormalization()(x_shortcut)

    x = layers.Add()([x, x_shortcut])

    x = layers.Activation('relu')(x)

    x = layers.Dropout(0.25)(x)



    return x



#Input Stage

inputs = layers.Input(shape=(28, 28, 1))

x = layers.Conv2D(filters=64, kernel_size=(4,4), 

                padding="Same", input_shape = (28, 28, 1))(inputs)

x = layers.BatchNormalization()(x)

x = layers.Activation('relu')(x)

x = layers.MaxPool2D(pool_size=(3,3))(x)



x = ConvBlock(x, filters=64)

x = IdentityBlock(x, filters=64)

x = IdentityBlock(x, filters=64)



x = ConvBlock(x, filters=128)

x = IdentityBlock(x, filters=128)

x = IdentityBlock(x, filters=128)



x = ConvBlock(x, filters=256)

x = IdentityBlock(x, filters=256)

x = IdentityBlock(x, filters=256)



#Final Stage

x = layers.AveragePooling2D(pool_size=(2,2))(x)

x = layers.GlobalAvgPool2D()(x)

x = layers.Dense(512, activation='relu')(x)

predictions = layers.Dense(10, activation='softmax')(x)

model = Model(inputs=inputs,outputs = predictions)

    

model.compile(optimizer="adam", 

              loss="sparse_categorical_crossentropy", 

              metrics = ["acc"])



model.summary()
history = model.fit_generator(datagen.flow(X_train,y_train, batch_size=256),

                              epochs = 75, validation_data = (X_valid,y_valid),steps_per_epoch = X_train.shape[0]/256,

                              verbose = 2, callbacks=callbacks_list)
pred_proba = model.predict(X_test)

y_pred = np.argmax(pred_proba, axis = 1)

sub['Label'] = y_pred
sub.to_csv("DigitRecogSub_MyResNet.csv")