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

"""
@author: Keras doc
https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
and fchollet
https://www.kaggle.com/fchollet/simple-deep-mlp-with-keras/code
"""

'''Trains a simple convnet on the MNIST dataset.
Original scripts gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU. (12x16s=3.2 minutes)
Geforce 1050: 11 seconds per epoch with batch size 128
Geforce 1050:  8 seconds per epoch with batch size 256
Geforce 1050:  6 seconds per epoch with batch size 512
Test loss: 0.023903
Test accuracy: 0.9923
'''

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
import matplotlib.pyplot as plt

batch_size = 128 #128
num_classes = 10
epochs = 60

# input image dimensions
img_rows, img_cols = 28, 28

# Read data
train = pd.read_csv('../input/train.csv')
labels = train.ix[:,0].values.astype('int32')
x_train = (train.ix[:,1:].values).astype('float32')
x_test = (pd.read_csv('../input/test.csv').values).astype('float32')

# https://www.codesofinterest.com/2017/09/keras-image-data-format.html
# Keras image_data_format():  channels_last
from keras import backend as K  
print("\n")
print("Keras image_data_format(): ",K.image_data_format())  

# reshape to (42000, 28, 28, 1)
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    # normalize
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print("\n")

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print("\n")
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(labels) 

# Built model with convolutional Layer and Dense layer and dropout
model = Sequential()
model.add(Conv2D(32, kernel_size=(8, 8),
                 activation='relu',
                 input_shape=input_shape))

model.add(Conv2D(64, (8, 8), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(num_classes, activation='softmax'))
earlystopping=[EarlyStopping(monitor='loss', patience=5, verbose=1, mode='auto')]

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_split=0.1,
          callbacks=earlystopping)

# predict the values
##score = model.evaluate(x_test, y_test, verbose=0)
##y_pred = model.predict(x_test).round()

# predict the values
print("Generating test predictions...")
preds = model.predict_classes(x_test, verbose=0)

def write_preds(preds, fname):
    pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)

write_preds(preds, "keras-cnn.csv")
print("Finished prediction.")
# plot CNN model (pip3 install pydot graphviz)
##from keras.utils import plot_model
##plot_model(model)
          
# plot learning curves

print("\n")
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Creating the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test.astype(bool).argmax(axis=1),y_pred.astype(bool).argmax(axis=1))
print(cm)

print("\n")
print('Test loss:', round(score[0],6))
print('Test accuracy:', score[1])
