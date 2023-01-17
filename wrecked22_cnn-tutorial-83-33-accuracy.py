import os

import numpy as np 

import pandas as pd 

import itertools

import tensorflow as tf 

from glob import glob

from PIL import Image

import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator, load_img

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix
test_path = '../input/seg_test/seg_test/'

train_path = '../input/seg_train/seg_train/'

pred_path = '../input/seg_pred/seg_pred/'
generate = ImageDataGenerator(rescale = 1./255,

                                   shear_range = 0.2,

                                   zoom_range = 0.2,

                                   horizontal_flip = True)
training_set = generate.flow_from_directory( train_path,

                                             target_size = (100, 100),

                                             batch_size = 14034,

                                             classes = ["buildings","forest","glacier","mountain","sea","street"],

                                             class_mode = 'categorical')

test_set = generate.flow_from_directory(     test_path,

                                             target_size = (100, 100),

                                             batch_size = 3000,

                                             classes = ["buildings","forest","glacier","mountain","sea","street"],

                                             class_mode = 'categorical')
X_train,y_train = training_set.next()

X_test,y_test = test_set.next()
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(128,(3,3), input_shape=(100,100,3), activation=tf.nn.relu, padding = "valid"))

model.add(tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=None))



model.add(tf.keras.layers.Conv2D(128,(3,3), activation=tf.nn.relu , padding= "same"))

model.add(tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=None))



model.add(tf.keras.layers.Conv2D(128,(3,3), activation=tf.nn.relu , padding= "same"))

model.add(tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=None))



model.add(tf.keras.layers.Flatten())



model.add(tf.keras.layers.Dense(128, activation= tf.nn.relu))

model.add(tf.keras.layers.Dropout(0.3))



model.add(tf.keras.layers.Dense(64, activation= tf.nn.relu))

model.add(tf.keras.layers.Dropout(0.25))



model.add(tf.keras.layers.Dense(6,activation = tf.nn.softmax))
model.compile(optimizer="adam", loss= "categorical_crossentropy", metrics=["accuracy"])
model.summary
Model = model.fit(X_train, y_train, epochs = 10, verbose=2, batch_size=32, validation_split = 0.1)
score = model.evaluate(X_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
plt.plot(Model.history['acc'])

plt.plot(Model.history['val_acc'])

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Training set', 'Validation set'], loc='upper left')

plt.show()
plt.plot(Model.history['val_loss'])

plt.plot(Model.history['loss'])

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Training set', 'Test set'], loc='upper left')

plt.show()
def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

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



y_pred = model.predict(X_test)

y_pred_classes = np.argmax(y_pred,axis = 1) 

y_true = np.argmax(y_test,axis = 1) 

confusion_mtx = confusion_matrix(y_true, y_pred_classes) 

plot_confusion_matrix(confusion_mtx, classes = range(6)) 