import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline

import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Convolution2D, MaxPooling2D

from keras.utils import np_utils

from keras.datasets import mnist

from keras import backend as K

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from sklearn.metrics import confusion_matrix

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

train.head()
trainY=train["label"]

trainX=train.drop(labels=["label"],axis=1) 

del train
g = sns.countplot(trainY)

trainY.value_counts()
trainX = trainX.values.reshape(trainX.shape[0], 1, 28, 28)

test = test.values.reshape(test.shape[0], 1, 28, 28)
trainX = trainX.astype('float32')

test = test.astype('float32')

trainX /= 255

test /= 255
# https://www.kaggle.com/adityaecdrid/mnist-with-keras-for-beginners-99457

plt.figure(figsize=(12,10))

x, y = 10, 4

for i in range(40):  

    plt.subplot(y, x, i+1)

    plt.imshow(trainX[i].reshape((28,28)),interpolation='nearest')

plt.show()

trainY = to_categorical(trainY, num_classes = 10)
from sklearn.model_selection import train_test_split

trainX, validX, trainY, validY = train_test_split(trainX, trainY, test_size = 0.1, random_state=42)
# https://elitedatascience.com/keras-tutorial-deep-learning-in-python

model = Sequential()

model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(1,28,28), data_format='channels_first'))

model.add(Convolution2D(32, 3, 3, activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))
model.summary()
from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau



# https://www.kaggle.com/adityaecdrid/mnist-with-keras-for-beginners-99457

model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.RMSprop(),

              metrics=['accuracy'])



learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.0001)



datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=15, # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images
datagen.fit(trainX)

h = model.fit_generator(datagen.flow(trainX,trainY, batch_size=32),

                              epochs = 10, validation_data = (validX, validY),

                              verbose = 1, steps_per_epoch=100

                              , callbacks=[learning_rate_reduction],)
# https://www.kaggle.com/adityaecdrid/mnist-with-keras-for-beginners-99457

# Look at confusion matrix 

import itertools

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

validPredY = model.predict(validX)

validPredYClasses = np.argmax(validPredY, axis = 1) 

validTrueY = np.argmax(validY, axis = 1) 

confusion_mtx = confusion_matrix(validTrueY, validPredYClasses) 

plot_confusion_matrix(confusion_mtx, classes = range(10))
print(h.history.keys())

import pylab

pylab.ylim([0,1])

accuracy = h.history['acc']

val_accuracy = h.history['val_acc']

loss = h.history['loss']

val_loss = h.history['val_loss']

epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'bo', label='Training accuracy')

plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend()

plt.show()

plt.figure()

pylab.ylim([0, 1])

plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()
