import numpy as np

import pandas as pd 

import os

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

import itertools 

import tensorflow as tf

import matplotlib.gridspec as gridspec 

from random import randint

from keras.utils.np_utils import to_categorical

from keras.layers import Conv2D,MaxPool2D,Dropout,Dense,Flatten,Activation

from keras.models import Sequential

from keras.optimizers import Adam

from keras.losses import categorical_crossentropy

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import RMSprop

from keras.callbacks import ReduceLROnPlateau

#Loading Librraries
test = pd.read_csv("../input/test.csv")

train = pd.read_csv("../input/train.csv")

labels = train.label

sns.countplot(labels)

plt.title('Categories');
train.drop("label",axis=1, inplace=True)
train = train / 255.0

test = test / 255.0

train = train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)

labels = to_categorical(labels)
g = plt.imshow(train[1][:,:,0]) # Must be a 0
X_train, X_test, Y_train, Y_test = train_test_split(train, labels, test_size=0.1, )
# model = tf.keras.models.Sequential()

# model.add(tf.keras.layers.Conv2D(32, (5,5), input_shape = (28,28,1) , activation= tf.nn.relu, padding= "valid"))

# model.add(tf.keras.layers.MaxPool2D(pool_size = (3,3), padding = "same",strides =(2,2)))

# model.add(tf.keras.layers.Conv2D(64, (5,5) , activation= tf.nn.relu, padding= "same"))

# model.add(tf.keras.layers.MaxPool2D(pool_size = (3,3), strides=(2,2)))

# model.add(tf.keras.layers.Flatten())

# model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))

# model.add(tf.keras.layers.Dropout(0.5))

# model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
# Set a learning rate annealer

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
epochs = 1 # Turn epochs to 30 to get 0.9967 accuracy

batch_size = 86
datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images





datagen.fit(X_train)
# Fit the model

history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (X_test,Y_test),

                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size

                              , callbacks=[learning_rate_reduction])

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=0)

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



y_pred = model.predict(X_test)

y_pred_classes = np.argmax(y_pred,axis = 1) 

y_true = np.argmax(Y_test,axis = 1) 

confusion_mtx = confusion_matrix(y_true, y_pred_classes) 

plot_confusion_matrix(confusion_mtx, classes = range(10)) 
predictions = model.predict(test)

label = np.argmax(predictions,axis = 1) 
test_pred = pd.DataFrame(model.predict(test))

test_pred = pd.DataFrame(test_pred.idxmax(axis = 1))

test_pred.index.name = 'ImageId'

test_pred = test_pred.rename(columns = {0: 'Label'}).reset_index()

test_pred['ImageId'] = test_pred['ImageId'] + 1



test_pred.head()
test_pred.to_csv('msubmission.csv', index = False)