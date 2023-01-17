import os

import numpy as np

import pandas as pd

import cv2

from IPython.display import Image, display 

from PIL import Image
print('the dictionaries present are', os.listdir('/kaggle/input/asl-rgb-depth-fingerspelling-spelling-it-out/dataset5/'))

dicts = os.listdir('/kaggle/input/asl-rgb-depth-fingerspelling-spelling-it-out/dataset5/')
y_train = []

X_train = []

y_val = []

X_val = []

y_test = []

X_test = []

path = '/kaggle/input/asl-rgb-depth-fingerspelling-spelling-it-out/dataset5/'+ dicts[0]

labels = os.listdir(path)

for label in labels:

    i = 0

    img_names = os.listdir(path+'/'+label)

    for img_name in img_names:

        if img_name.startswith('color'):

            img = cv2.imread(path+'/'+label+ '/'+img_name)

            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            img_resized = cv2.resize(img_gray, (32,32))

            X_train.append(np.array(img_resized))

            y_train.append(label)

            i = i+1

            if i > 750:

                break

path = '/kaggle/input/asl-rgb-depth-fingerspelling-spelling-it-out/dataset5/'+ dicts[1]

labels = os.listdir(path)

for label in labels:

    i = 0

    img_names = os.listdir(path+'/'+label)

    for img_name in img_names:

        if img_name.startswith('color'):

            img = cv2.imread(path+'/'+label+ '/'+img_name)

            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            img_resized = cv2.resize(img_gray, (32,32))

            X_val.append(np.array(img_resized))

            y_val.append(label)

            i = i+1

            if i > 250:

                break

path = '/kaggle/input/asl-rgb-depth-fingerspelling-spelling-it-out/dataset5/'+ dicts[2]

labels = os.listdir(path)

for label in labels:

    i = 0

    img_names = os.listdir(path+'/'+label)

    for img_name in img_names:

        if img_name.startswith('color'):

            img = cv2.imread(path+'/'+label+ '/'+img_name)

            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            img_resized = cv2.resize(img_gray, (32,32))

            X_test.append(np.array(img_resized))

            y_test.append(label)

            i = i+1

            if i >250:

                break
X_train = np.array(X_train)

y_train = np.array(y_train)

X_val = np.array(X_val)

y_val = np.array(y_val)

X_test = np.array(X_test)

y_test = np.array(y_test)

print('the sizes of train data are:', X_train.shape, 'and', y_train.shape, '\n', 'the sizes of test data are',

      X_test.shape, 'and', y_test.shape, '\n', 'the sizes of validation data are:', X_val.shape, 'and', y_val.shape)
print('the number of labels in y_train', len(np.unique(y_train)), '\n',

     'the number of labels in y_test', len(np.unique(y_test)), '\n',

     'the number of labels in y_val', len(np.unique(y_val)))
X_train_std = X_train.astype('float32')/255

X_test_std =X_test.astype('float32')/255

X_val_std = X_val.astype('float32')/255
X_train_std.shape[1:]
from sklearn.preprocessing import LabelEncoder

from keras.utils import to_categorical

y_train_coded = LabelEncoder().fit_transform(y_train)

y_train_coded = to_categorical(y_train_coded)

y_test_coded = LabelEncoder().fit_transform(y_test)

y_test_coded = to_categorical(y_test_coded)

y_val_coded = LabelEncoder().fit_transform(y_val)

y_val_coded = to_categorical(y_val_coded)
from keras import models, layers
#using max pooling(size = 2,2) subsampling layers:

model_maxpool_2 = models.Sequential()

model_maxpool_2.add(layers.Conv2D(filters = 32, kernel_size = (5,5), activation = 'relu', input_shape = (32,32,1)))

model_maxpool_2.add(layers.MaxPool2D(pool_size = (2,2)))

model_maxpool_2.add(layers.Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))

model_maxpool_2.add(layers.MaxPool2D(pool_size = (2,2)))

model_maxpool_2.add(layers.Dropout(rate = 0.25))

model_maxpool_2.add(layers.Flatten())

model_maxpool_2.add(layers.Dense(256, activation = 'relu'))

model_maxpool_2.add(layers.Dropout(rate = 0.5))

model_maxpool_2.add(layers.Dense(24, activation = 'softmax'))
# compile the model

model_maxpool_2.compile(loss = 'categorical_crossentropy',

                        optimizer = 'adam',

                        metrics = ['accuracy'])
#let's take only 2000 samples of train data, 500 samples of val data and 200 samples of test data

r = np.arange(X_train_std.shape[0])

np.random.seed(42)

np.random.shuffle(r)

X_train = X_train_std[r]

X_train_data = X_train[:2000,:,:]

X_train_data = X_train_data.reshape((2000, 32, 32, 1))

y_train = y_train_coded[r]

y_train_data = y_train[:2000]

X_test_data = X_train[2001:2501,:,:]

X_test_data = X_test_data.reshape((500, 32, 32, 1))

y_test_data = y_train[2001:2501]

X_val_data = X_train[2501:3001,:,:]

X_val_data = X_val_data.reshape((500, 32, 32, 1))

y_val_data = y_train[3001:3501]
#fitting the model

history = model_maxpool_2.fit(X_train_data, y_train_data, epochs=25, validation_data=(X_val_data, y_val_data))
#Display of the accuracy and the loss values

import matplotlib.pyplot as plt



plt.figure(figsize = (8,8))

plt.plot(history.history['accuracy'], label='training accuracy')

plt.plot(history.history['val_accuracy'], label='val accuracy')

plt.plot(history.history['loss'], label='training loss')

plt.plot(history.history['val_loss'], label='val loss')

plt.title('Loss/accuracy')

plt.xlabel('epochs')

plt.ylabel('loss')

plt.legend()

plt.show()
y_pred_maxpool_2 = model_maxpool_2.predict_classes(X_test_data)

y_test_maxpool_2 = np.argmax(y_test_data, axis = 1)

from sklearn.metrics import accuracy_score

accuracy_maxpool_2 = accuracy_score(y_pred_maxpool_2,y_test_maxpool_2)

print('the accuracy obtained on the test set is:', accuracy_maxpool_2)
from sklearn.metrics import classification_report

print(classification_report(y_test_maxpool_2, y_pred_maxpool_2))
#Let's see how average pooling works on the data instead of maxpooling

model_avgpool_2 = models.Sequential()

model_avgpool_2.add(layers.Conv2D(filters = 32, kernel_size = (5,5), activation = 'relu', input_shape = (32,32,1)))

model_avgpool_2.add(layers.AveragePooling2D(pool_size = (2,2)))

model_avgpool_2.add(layers.Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))

model_avgpool_2.add(layers.AveragePooling2D(pool_size = (2,2)))

model_avgpool_2.add(layers.Dropout(rate = 0.25))

model_avgpool_2.add(layers.Flatten())

model_avgpool_2.add(layers.Dense(256, activation = 'relu'))

model_avgpool_2.add(layers.Dropout(rate = 0.5))

model_avgpool_2.add(layers.Dense(24, activation = 'softmax'))
# compile the model

model_avgpool_2.compile(loss = 'categorical_crossentropy',

                        optimizer = 'adam',

                        metrics = ['accuracy'])
#fitting the model

history_avgpool_2 = model_avgpool_2.fit(X_train_data, y_train_data, epochs=25, validation_data=(X_val_data, y_val_data))
#Display of the accuracy and the loss values

plt.figure(figsize = (8,8))

plt.plot(history_avgpool_2.history['accuracy'], label='training accuracy')

plt.plot(history_avgpool_2.history['val_accuracy'], label='val accuracy')

plt.plot(history_avgpool_2.history['loss'], label='training loss')

plt.plot(history_avgpool_2.history['val_loss'], label='val loss')

plt.title('Loss/accuracy')

plt.xlabel('epochs')

plt.ylabel('loss')

plt.legend()

plt.show()
y_pred_avgpool_2 = model_avgpool_2.predict_classes(X_test_data)

y_test_avgpool_2 = np.argmax(y_test_data, axis = 1)

from sklearn.metrics import accuracy_score

accuracy_avgpool_2 = accuracy_score(y_pred_avgpool_2,y_test_avgpool_2)

print('the accuracy obtained on the test set is:', accuracy_avgpool_2)
from sklearn.metrics import classification_report

print(classification_report(y_test_avgpool_2, y_pred_avgpool_2))
#What if the kernel size in the convolution layers is increased?

model_kernel_up = models.Sequential()

model_kernel_up.add(layers.Conv2D(filters = 32, kernel_size = (8,8), activation = 'relu', input_shape = (32,32,1)))

model_kernel_up.add(layers.AveragePooling2D(pool_size = (2,2)))

model_kernel_up.add(layers.Conv2D(filters = 64, kernel_size = (5,5), activation = 'relu'))

model_kernel_up.add(layers.AveragePooling2D(pool_size = (2,2)))

model_kernel_up.add(layers.Dropout(rate = 0.25))

model_kernel_up.add(layers.Flatten())

model_kernel_up.add(layers.Dense(256, activation = 'relu'))

model_kernel_up.add(layers.Dropout(rate = 0.5))

model_kernel_up.add(layers.Dense(24, activation = 'softmax'))
# compile the model

model_kernel_up.compile(loss = 'categorical_crossentropy',

                        optimizer = 'adam',

                        metrics = ['accuracy'])
#fitting the model

history_kernel_up = model_kernel_up.fit(X_train_data, y_train_data, epochs=25, validation_data=(X_val_data, y_val_data))
#Display of the accuracy and the loss values

plt.figure(figsize = (8,8))

plt.plot(history_kernel_up.history['accuracy'], label='training accuracy')

plt.plot(history_kernel_up.history['val_accuracy'], label='val accuracy')

plt.plot(history_kernel_up.history['loss'], label='training loss')

plt.plot(history_kernel_up.history['val_loss'], label='val loss')

plt.title('Loss/accuracy')

plt.xlabel('epochs')

plt.ylabel('loss')

plt.legend()

plt.show()
y_pred_kernel_up = model_kernel_up.predict_classes(X_test_data)

y_test_kernel_up = np.argmax(y_test_data, axis = 1)

from sklearn.metrics import accuracy_score

accuracy_kernel_up = accuracy_score(y_pred_kernel_up,y_test_kernel_up)

print('the accuracy obtained on the test set is:', accuracy_kernel_up)
from sklearn.metrics import classification_report

print(classification_report(y_test_kernel_up, y_pred_kernel_up))
#what the effect if the pooling size is increased?



model_avgpool_5 = models.Sequential()

model_avgpool_5.add(layers.Conv2D(filters = 32, kernel_size = (5,5), activation = 'relu', input_shape = (32,32,1)))

model_avgpool_5.add(layers.AveragePooling2D(pool_size = (3,3)))

model_avgpool_5.add(layers.Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))

model_avgpool_5.add(layers.AveragePooling2D(pool_size = (3,3)))

model_avgpool_5.add(layers.Dropout(rate = 0.25))

model_avgpool_5.add(layers.Flatten())

model_avgpool_5.add(layers.Dense(256, activation = 'relu'))

model_avgpool_5.add(layers.Dropout(rate = 0.5))

model_avgpool_5.add(layers.Dense(24, activation = 'softmax'))
# compile the model

model_avgpool_5.compile(loss = 'categorical_crossentropy',

                        optimizer = 'adam',

                        metrics = ['accuracy'])
#fitting the model

history_avgpool_5 = model_avgpool_5.fit(X_train_data, y_train_data, epochs=25, validation_data=(X_val_data, y_val_data))
#Display of the accuracy and the loss values

plt.figure(figsize = (8,8))

plt.plot(history_avgpool_5.history['accuracy'], label='training accuracy')

plt.plot(history_avgpool_5.history['val_accuracy'], label='val accuracy')

plt.plot(history_avgpool_5.history['loss'], label='training loss')

plt.plot(history_avgpool_5.history['val_loss'], label='val loss')

plt.title('Loss/accuracy')

plt.xlabel('epochs')

plt.ylabel('loss')

plt.legend()

plt.show()
y_pred_avgpool_5 = model_avgpool_5.predict_classes(X_test_data)

y_test_avgpool_5 = np.argmax(y_test_data, axis = 1)

from sklearn.metrics import accuracy_score

accuracy_avgpool_5 = accuracy_score(y_pred_avgpool_5,y_test_avgpool_5)

print('the accuracy obtained on the test set is:', accuracy_avgpool_5)
from sklearn.metrics import classification_report

print(classification_report(y_test_avgpool_5, y_pred_avgpool_5))