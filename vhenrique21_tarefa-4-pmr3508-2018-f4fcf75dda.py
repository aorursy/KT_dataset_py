# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D as MaxPool2D
from keras.callbacks import EarlyStopping
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
labels = pd.read_csv("../input/train_labels.csv",index_col=0)
labels.shape
labels_categorical = np_utils.to_categorical(labels)
original = np.load('../input/train_images_pure.npy')
original.shape
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(original[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[labels.label[i]])
rotated = np.load('../input/train_images_rotated.npy')
rotated.shape
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(rotated[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[labels.label[i]])
noisy = np.load('../input/train_images_noisy.npy')
noisy.shape
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(noisy[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[labels.label[i]])
both = np.load('../input/train_images_both.npy')
both.shape
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(both[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[labels.label[i]])
test = np.load('../input/Test_images.npy')
test.shape
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test[i], cmap=plt.cm.binary)
plt.figure()
plt.imshow(both[0])
plt.colorbar()
plt.grid(False)
original = original / 255.0
rotated = rotated / 255.0
noisy = noisy / 255.0
both = both / 255.0

test = test / 255.0
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(both[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[labels.label[i]])
seed = 7
np.random.seed(seed)
Xoriginal = original.reshape(original.shape[0], 28, 28, 1).astype('float32')
Xnoisy = noisy.reshape(noisy.shape[0], 28, 28, 1).astype('float32')
Xrotated = rotated.reshape(rotated.shape[0], 28, 28, 1).astype('float32')
Xboth = both.reshape(both.shape[0], 28, 28, 1).astype('float32')

XtestFinal = test.reshape(test.shape[0], 28, 28, 1).astype('float32')
def baseline_model():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    #model.add(Conv2D(64, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
modelo = baseline_model()
modelo.summary()
callbacks = [EarlyStopping(monitor = 'val_loss', patience = 2)]
Xtrain,Xtest,Ytrain,Ytest = train_test_split(Xoriginal,labels_categorical, test_size = 0.25)
modelo.fit(Xtrain, Ytrain, validation_data=(Xtest,Ytest), epochs=40, 
          batch_size = 200, verbose=1, callbacks = callbacks)
#Dados de Validação:
scores = modelo.evaluate(Xtest, Ytest, verbose=0)
print("Validação:", scores[1])

#Dados Originais:
scores = modelo.evaluate(Xoriginal, labels_categorical, verbose=0)
print("Original:", scores[1])

#Dados Rotated:
scores = modelo.evaluate(Xrotated, labels_categorical, verbose=0)
print("Rotated:", scores[1])

#Dados Noisy:
scores = modelo.evaluate(Xnoisy, labels_categorical, verbose=0)
print("Noisy:", scores[1])

#Dados Both:
scores = modelo.evaluate(Xboth, labels_categorical, verbose=0)
print("Both:", scores[1])
Xtrain,Xtest,Ytrain,Ytest = train_test_split(Xboth,labels_categorical, test_size = 0.25)
modelo.fit(Xtrain, Ytrain, validation_data=(Xtest,Ytest), epochs=40, 
          batch_size = 200, verbose=1, callbacks = callbacks)
#Dados de Validação:
scores = modelo.evaluate(Xtest, Ytest, verbose=0)
print("Validação:", scores[1])

#Dados Originais:
scores = modelo.evaluate(Xoriginal, labels_categorical, verbose=0)
print("Original:", scores[1])

#Dados Rotated:
scores = modelo.evaluate(Xrotated, labels_categorical, verbose=0)
print("Rotated:", scores[1])

#Dados Noisy:
scores = modelo.evaluate(Xnoisy, labels_categorical, verbose=0)
print("Noisy:", scores[1])

#Dados Both:
scores = modelo.evaluate(Xboth, labels_categorical, verbose=0)
print("Both:", scores[1])
seed = 7
np.random.seed(seed)
Xoriginal = original.reshape(original.shape[0], 28, 28, 1).astype('float32')
Xnoisy = noisy.reshape(noisy.shape[0], 28, 28, 1).astype('float32')
Xrotated = rotated.reshape(rotated.shape[0], 28, 28, 1).astype('float32')
Xboth = both.reshape(both.shape[0], 28, 28, 1).astype('float32')

XtestFinal = test.reshape(test.shape[0], 28, 28, 1).astype('float32')
def baseline_model():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), padding="same", input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPool2D((2,2)))
    model.add(Conv2D(64, (5, 5), padding="same", activation='relu'))
    model.add(MaxPool2D((2,2)))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
modeloPool = baseline_model()
modeloPool.summary()
callbacks = [EarlyStopping(monitor = 'val_loss', patience = 2)]
Xtrain,Xtest,Ytrain,Ytest = train_test_split(Xoriginal,labels_categorical, test_size = 0.25)
modeloPool.fit(Xtrain, Ytrain, validation_data=(Xtest,Ytest), epochs=40, 
          batch_size = 200, verbose=1, callbacks = callbacks)
#Dados de Validação:
scores = modeloPool.evaluate(Xtest, Ytest, verbose=0)
print("Validação:", scores[1])

#Dados Originais:
scores = modeloPool.evaluate(Xoriginal, labels_categorical, verbose=0)
print("Original:", scores[1])

#Dados Rotated:
scores = modeloPool.evaluate(Xrotated, labels_categorical, verbose=0)
print("Rotated:", scores[1])

#Dados Noisy:
scores = modeloPool.evaluate(Xnoisy, labels_categorical, verbose=0)
print("Noisy:", scores[1])

#Dados Both:
scores = modeloPool.evaluate(Xboth, labels_categorical, verbose=0)
print("Both:", scores[1])
Xtrain,Xtest,Ytrain,Ytest = train_test_split(Xboth,labels_categorical, test_size = 0.25)
modeloPool.fit(Xtrain, Ytrain, validation_data=(Xtest,Ytest), epochs=40, 
          batch_size = 200, verbose=1, callbacks = callbacks)
#Dados de Validação:
scores = modeloPool.evaluate(Xtest, Ytest, verbose=0)
print("Validação:", scores[1])

#Dados Originais:
scores = modeloPool.evaluate(Xoriginal, labels_categorical, verbose=0)
print("Original:", scores[1])

#Dados Rotated:
scores = modeloPool.evaluate(Xrotated, labels_categorical, verbose=0)
print("Rotated:", scores[1])

#Dados Noisy:
scores = modeloPool.evaluate(Xnoisy, labels_categorical, verbose=0)
print("Noisy:", scores[1])

#Dados Both:
scores = modeloPool.evaluate(Xboth, labels_categorical, verbose=0)
print("Both:", scores[1])
Pred = modeloPool.predict_classes(XtestFinal)

result = pd.DataFrame(columns = ['Id','label'])
result.label = Pred
result.Id = range(len(test))
result.to_csv("result.csv",index=False)