import keras

import keras.utils

import sklearn

from sklearn.model_selection import cross_validate

import sklearn.metrics

import matplotlib.pyplot as plt

from keras.utils import to_categorical

from keras import utils as np_utils

from keras.models import Sequential

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam

from keras.layers import Dense, Dropout, Activation, MaxPool2D, Flatten

from keras.layers.convolutional import Conv2D

from keras.layers.normalization import BatchNormalization

from keras.optimizers import SGD

import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

import numpy as np

import cv2

import os
DATASET_PATH = '../input/15-scene/15-Scene/'



one_hot_lookup = np.eye(15) # 15 classes



dataset_x = []

dataset_y = []



for category in sorted(os.listdir(DATASET_PATH)):

    print('loading category: '+str(int(category)))

    for fname in os.listdir(DATASET_PATH+category):

        img = cv2.imread(DATASET_PATH+category+'/'+fname, 2)

        img = cv2.resize(img, (224,224))

        dataset_x.append(np.reshape(img, [224,224,1]))

        dataset_y.append(np.reshape(one_hot_lookup[int(category)], [15]))



dataset_x = np.array(dataset_x)

dataset_y = np.array(dataset_y)





"""shuffle dataset"""

p = np.random.permutation(len(dataset_x))

dataset_x = dataset_x[p]

dataset_y = dataset_y[p]

        

X_test = dataset_x[:int(len(dataset_x)/10)]

Y_test = dataset_y[:int(len(dataset_x)/10)]

X_train = dataset_x[int(len(dataset_x)/10):]

Y_train = dataset_y[int(len(dataset_x)/10):]
print(train_x.shape)

print(train_y.shape)

print(test_x.shape)

print(test_y.shape)
def kerasCustom(a,b,c,d):

    batch_size = 128

    num_classes = 15

    epochs = 100

    img_rows, img_cols = X_train.shape[1],X_train.shape[2]

    input_shape = (img_rows, img_cols, 1)

    model = Sequential()

    model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 

                     activation ='relu', input_shape = input_shape))

    model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 

                     activation ='relu'))

    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(BatchNormalization())

    model.add(Dropout(0.25))

    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                     activation ='relu'))

    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                     activation ='relu'))

    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(BatchNormalization())

    model.add(Dropout(0.25))

    model.add(Conv2D(filters = 86, kernel_size = (3,3),padding = 'Same', 

                     activation ='relu'))

    model.add(Conv2D(filters = 86, kernel_size = (3,3),padding = 'Same', 

                     activation ='relu'))

    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(BatchNormalization())

    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(1024, activation = "relu"))

    model.add(Dropout(0.5))

    model.add(Dense(512, activation = "relu"))

    model.add(Dropout(0.5))

    #model.add(Flatten())

    model.add(Dense(15, activation = "softmax"))

    # Define the optimizer

    optimizer = Adam(lr=0.0005, decay=1e-6)

    model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

    datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        shear_range=0.2,

        zoom_range=0.2,

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=True,  # randomly flip images

        vertical_flip=False)  # randomly flip images

    datagen.fit(a)

    history = model.fit_generator(datagen.flow(a,b, batch_size=64),steps_per_epoch=len(a) / 64, epochs=epochs)

    score = model.evaluate(c,d, verbose=1)

    print('\nAccuracy:', score[1])

    # Plot training & validation accuracy values

    print('Plot training & validation accuracy values\n')

    plt.plot(history.history['acc'])

    plt.plot(history.history['val_acc'])

    plt.title('Model accuracy')

    plt.ylabel('Accuracy')

    plt.xlabel('Epoch')

    plt.legend(['Train', 'Test'], loc='upper left')

    plt.show()

    

    # Plot training & validation loss values

    print('Plot training & validation loss values\n')

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('Model loss')

    plt.ylabel('Loss')

    plt.xlabel('Epoch')

    plt.legend(['Train', 'Test'], loc='upper left')

    plt.show()

    

kerasCustom(X_train, Y_train, X_test, Y_test)