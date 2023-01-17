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
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
%matplotlib inline
data_train = pd.read_csv("../input/train.csv")
data_test = pd.read_csv("../input/test.csv")

img_rows = 28
img_cols = 28
input_shape = (img_rows, img_cols, 1)
X = np.array(data_train.iloc[10, 1:])
plt.imshow(X.reshape([28,28]))

X = np.array(data_train.iloc[:, 1:])
y = to_categorical(np.array(data_train.iloc[:, 0]))

X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.4, random_state=13)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=13)
for xx in (X_train, X_val, X_test):
    print(xx.shape)
    xx.shape = (xx.shape[0], img_rows, img_rows, 1)
    
print(X_train.shape, X_test.shape, X_test.shape)

plt.subplot(311)
plt.hist(y_train.argmax(1))
plt.subplot(312)
plt.hist(y_val.argmax(1))
plt.subplot(313)
plt.hist(y_test.argmax(1))

X_train = X_train.astype(np.float32)
X_val   = X_val.astype(np.float32)
X_test  = X_test.astype(np.float32)
X_train = X_train/255
X_val   = X_val/255
X_test  = X_test/255
X_train.dtype

import keras
from keras.models import Sequential
from keras.layers import Dropout, Conv2D, MaxPooling2D, Flatten, Activation
from keras.layers.normalization import BatchNormalization
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau

from keras import backend as K
print(K.image_data_format())
print(K.tensorflow_backend._get_available_gpus())

batch_size = 256
num_classes = 10

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), padding='same', kernel_initializer='he_normal', input_shape=input_shape))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(32, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(64, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(32, kernel_size=(5,5), padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(num_classes, kernel_size=(3,3), padding='valid'))

model.add(Flatten())

model.add(Activation('softmax'))
model.summary()

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])

epochs = 50
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=0.00001)
history = model.fit(X_train, y_train, batch_size=batch_size, callbacks=[reduce_lr], epochs=epochs, verbose=1, validation_data=(X_val, y_val))
score = model.evaluate(X_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


%matplotlib inline
accuracy = history.history['acc']
val_accuracy = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

predicted_classes = model.predict_classes(X_test)
y_true = y_test.argmax(1)
correct = np.nonzero(predicted_classes==y_true)[0]
incorrect = np.nonzero(predicted_classes!=y_true)[0]
from sklearn.metrics import classification_report
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(y_true, predicted_classes, target_names=target_names))




for i, incorrect in enumerate(incorrect[0:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_true[incorrect]))
    plt.tight_layout()