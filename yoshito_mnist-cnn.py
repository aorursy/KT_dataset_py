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
train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")
train_data.head()
test_data.head()
# train data

x_name = []

for i in range(28*28):

    x_name.append("pixel"+str(i))



X_train = train_data[x_name]

X_train = np.array(X_train, dtype=np.float32)

X_train = X_train.reshape([-1, 28, 28, 1])

X_train = X_train / 127.5 - 1



_y_train = train_data["label"]

_y_train = np.array(_y_train)

y_train = np.zeros([_y_train.shape[0], 10])

y_train[np.arange(_y_train.shape[0]), _y_train] = 1



# test data

X_test = test_data[x_name]

X_test = np.array(X_test, dtype=np.float32)

X_test = X_test.reshape([-1, 28, 28, 1])

X_test = X_test / 127.5 - 1



print("train data", X_train.shape, y_train.shape)

print("test data", X_test.shape)
import keras

from keras.layers import Activation, Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout, BatchNormalization

from keras.models import Model



def Module():

    inputs = Input((28, 28, 1))

    x = Conv2D(32, [3,3], padding="same", strides=1, activation=None)(inputs)

    x = BatchNormalization()(x)

    x = Activation("relu")(x)

    x = Conv2D(32, [3,3], padding="same", strides=1, activation=None)(x)

    x = BatchNormalization()(x)

    x = Activation("relu")(x)

    x = MaxPooling2D([2,2], strides=2)(x)

    x = Conv2D(64, [3,3], padding="same", strides=1, activation=None)(x)

    x = BatchNormalization()(x)

    x = Activation("relu")(x)

    x = Conv2D(64, [3,3], padding="same", strides=1, activation=None)(x)

    x = BatchNormalization()(x)

    x = Activation("relu")(x)

    x = MaxPooling2D([2, 2], strides=2)(x)

    x = Flatten()(x)

    x = Dense(4096, activation='relu')(x)

    #x = Dropout(rate=0.5)(x)

    x = Dense(4096, activation='relu')(x)

    #x = Dropout(rate=0.5)(x)

    x = Dense(4096, activation='relu')(x)

    #x = Dropout(rate=0.5)(x)

    x = Dense(10, activation="softmax")(x)

    module = Model(inputs=inputs, outputs=x)

    return module



model = Module()

model.compile(

        loss='categorical_crossentropy',

        optimizer=keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True),

        #optimizer=keras.optimizers.Adam(lr=0.01),

        metrics=['accuracy'])



history = model.fit(X_train, y_train, epochs=100, batch_size=512)



model.compile(

        loss='categorical_crossentropy',

        optimizer=keras.optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True),

        metrics=['accuracy'])



history = model.fit(X_train, y_train, epochs=100, batch_size=512)



model.compile(

        loss='categorical_crossentropy',

        optimizer=keras.optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True),

        metrics=['accuracy'])



history = model.fit(X_train, y_train, epochs=100, batch_size=512)



#from keras.preprocessing.image import ImageDataGenerator



#train_datagen = ImageDataGenerator(

        #rescale=1. / 255,

#        shear_range=0.2,

#        zoom_range=0.2,

#        horizontal_flip=True

        #featurewise_center=True

        #featurewise_std_normalization=True,

        #zca_whitening=True

#    )



#train_datagen.fit(X_train)

#train_generator = train_datagen.flow(X_train,y_train, batch_size=512,)



#history = model.fit_generator(train_generator, steps_per_epoch=512, epochs=100)

preds = model.predict(X_test)

preds = preds.argmax(axis=-1)
preds_submit = pd.DataFrame([np.arange(1, len(preds)+1), preds]).T

preds_submit.columns = ["ImageId", "Label"]

preds_submit.to_csv("kaggle_mnist_submission.csv",index=False)