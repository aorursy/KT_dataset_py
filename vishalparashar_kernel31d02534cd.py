# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import sys
import csv
import os
import math
import json, codecs
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from zipfile import ZipFile
import shutil
from glob import glob
from PIL import Image
from PIL import ImageFilter
from sklearn.model_selection import train_test_split
import keras
from keras import backend as K
from keras import layers
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential, load_model
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
import torch 
import torch.nn as nn
import torch.nn.functional as F
import time
import cv2
import h5py

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_path = "/kaggle/input/signature-recognition-mini/signatures_10/train"
test_path = "/kaggle/input/signature-recognition-mini/signatures_10/test"
test_g = []
test_f = []
for dirname, _, filenames in os.walk(test_path+'/genuine'):
    for filename in filenames:
        test_g.append(os.path.join(dirname, filename))

for dirname, _, filenames in os.walk(test_path+'/forged'):
    for filename in filenames:
        test_f.append(os.path.join(dirname, filename))

train_g = []
train_f = []
for dirname, _, filenames in os.walk(train_path+'/genuine'):
    for filename in filenames:
        train_g.append(os.path.join(dirname, filename))

for dirname, _, filenames in os.walk(train_path+'/forged'):
    for filename in filenames:
        train_f.append(os.path.join(dirname, filename))
train_g = ['/kaggle/input/signature/signature/Train/genuine/g40.png',
'/kaggle/input/signature/signature/Train/genuine/g128.PNG',
'/kaggle/input/signature/signature/Train/genuine/g3.png',
'/kaggle/input/signature/signature/Train/genuine/g42.png',
'/kaggle/input/signature/signature/Train/genuine/g81.png',
'/kaggle/input/signature/signature/Train/genuine/g120.png',
'/kaggle/input/signature/signature/Train/genuine/g83.png',
'/kaggle/input/signature/signature/Train/genuine/g131.PNG',
'/kaggle/input/signature/signature/Train/genuine/g145.PNG',
'/kaggle/input/signature/signature/Train/genuine/g56.png']

train_f = ['/kaggle/input/signature/signature/Train/forged/f27.PNG',
'/kaggle/input/signature/signature/Train/forged/f55.png',
'/kaggle/input/signature/signature/Train/forged/f140.png',
'/kaggle/input/signature/signature/Train/forged/f68.png',
'/kaggle/input/signature/signature/Train/forged/f26.png',
'/kaggle/input/signature/signature/Train/forged/f135.png',
'/kaggle/input/signature/signature/Train/forged/f33.png',
'/kaggle/input/signature/signature/Train/forged/f80.png',
'/kaggle/input/signature/signature/Train/forged/f117.png',
'/kaggle/input/signature/signature/Train/forged/f114.png']

test_g = ['/kaggle/input/signature/signature/Test/genuine/g4.png',
'/kaggle/input/signature/signature/Test/genuine/g59.png',
'/kaggle/input/signature/signature/Test/genuine/g12.png',
'/kaggle/input/signature/signature/Test/genuine/g35.png',
'/kaggle/input/signature/signature/Test/genuine/g13.png',
'/kaggle/input/signature/signature/Test/genuine/g21.png',
'/kaggle/input/signature/signature/Test/genuine/g24.png',
'/kaggle/input/signature/signature/Test/genuine/g25.png',
'/kaggle/input/signature/signature/Test/genuine/g19.png',
'/kaggle/input/signature/signature/Test/genuine/g14.png']

test_f = ['/kaggle/input/signature/signature/Test/forge/f55.png',
'/kaggle/input/signature/signature/Test/forge/f26.png',
'/kaggle/input/signature/signature/Test/forge/f12.png',
'/kaggle/input/signature/signature/Test/forge/f33.png',
'/kaggle/input/signature/signature/Test/forge/f4.png',
'/kaggle/input/signature/signature/Test/forge/f53.png',
'/kaggle/input/signature/signature/Test/forge/f17.png',
'/kaggle/input/signature/signature/Test/forge/f48.png',
'/kaggle/input/signature/signature/Test/forge/f10.png',
'/kaggle/input/signature/signature/Test/forge/f44.png']
train_g
plt.figure(figsize = (35,5))
plt.suptitle('Train Real Signatures', fontsize = 18)
x, y = 1, 10
for i in range(10):
    plt.subplot(x, y, i+1)
    plt.axis('off')
    plt.imshow(cv2.resize(cv2.imread(train_g[i], 1), (224,224)))
plt.savefig('train_g')
plt.show()    
plt.figure(figsize = (35,5))
plt.suptitle('Train Fake Signatures', fontsize = 18)
x, y = 1, 10
for i in range(10):
    plt.subplot(x, y, i+1)
    plt.axis('off')
    plt.imshow(cv2.resize(cv2.imread(train_f[i], 1), (224,224)))
plt.savefig('train_f')
plt.show()
plt.figure(figsize = (35,5))
plt.suptitle('Test Real Signatures', fontsize = 18)
x, y = 1, 10
for i in range(10):
    plt.subplot(x, y, i+1)
    plt.axis('off')
    plt.imshow(cv2.resize(cv2.imread(test_g[i], 1), (224,224)))
plt.savefig('test_g')
plt.show()    
plt.figure(figsize = (35,5))
plt.suptitle('Test Fake Signatures', fontsize = 18)
x, y = 1, 10
for i in range(10):
    plt.subplot(x, y, i+1)
    plt.axis('off')
    plt.imshow(cv2.resize(cv2.imread(test_f[i], 1), (224,224)))
plt.savefig('test_f')
plt.show()    
numberOfClass = len(glob(train_path+"/*"))
train_data = ImageDataGenerator().flow_from_directory(train_path, target_size=(224,224), class_mode='binary')
test_data = ImageDataGenerator().flow_from_directory(test_path, target_size=(224,224), class_mode='binary')
# Data replication

train_datagen = ImageDataGenerator(
    shear_range=10,
    zoom_range=0.2,
    horizontal_flip=True,
    preprocessing_function=preprocess_input)
 
train_generator = train_datagen.flow_from_directory(
    train_path,
    batch_size=32,
    class_mode='binary',
    target_size=(224,224))
 
test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input)
 
test_generator = test_datagen.flow_from_directory(
    test_path,
    shuffle=False,
    class_mode='binary',
    target_size=(224,224))
epochs = [5,10,15]
optimizers = ['SGD', 'Adam', 'RMSprop']
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(224,224,3)))
model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.50))

model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.50))


model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.50))
model.add(Dense(numberOfClass, activation='softmax'))


print(model.summary())
from keras.utils import plot_model
plot_model(model)
for epoch in epochs:
    print("Epoch:", epoch)
    for optimizer in optimizers:
        model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
        
        
        print("Optimizer:",optimizer)
        history = model.fit_generator(
            generator=train_data,
            epochs=epoch,
            validation_data=test_data,
            steps_per_epoch=40,
            validation_steps=20)
        
        val_loss, val_acc = model.evaluate(test_data)
        print(val_loss)
        print(val_acc)
        model.save('cnn_0_num_reader.model')
        
        plt.figure(figsize=(30,5))
        plt.subplot(121)
        title = 'Model:CNN Epoch:'+str(epoch)+' Optimizer:'+optimizer+' Data Replication: False'
        plt.suptitle(title, fontsize=15)


        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy', fontsize=10)
        plt.ylabel('Accuracy', fontsize=10)
        plt.xlabel('Epoch', fontsize=10)
        plt.legend(['Train', 'Test'], loc='upper left')


        plt.subplot(122)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss', fontsize=10)
        plt.ylabel('Loss', fontsize=10)
        plt.xlabel('Epoch', fontsize=10)
        plt.legend(['Train', 'Test'], loc='upper left')

        save = "cnn_0_"+optimizer+"_"+str(epoch)
        plt.savefig(save)
        plt.show()
for epoch in epochs:
    print("Epoch:", epoch)
    for optimizer in optimizers:
        model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
               
        print("Optimizer:",optimizer)
        history = model.fit_generator(
            generator=train_generator,
            epochs=epoch,
            validation_data=test_generator,
            steps_per_epoch=40,
            validation_steps=20)
        
        val_loss, val_acc = model.evaluate(test_data)
        print(val_loss)
        print(val_acc)
        model.save('cnn_1_num_reader.model')
        
        plt.figure(figsize=(30,5))
        plt.subplot(121)
        title = 'Model:CNN Epoch:'+str(epoch)+' Optimizer:'+optimizer+' Data Replication: True'
        plt.suptitle(title, fontsize=15)


        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy', fontsize=10)
        plt.ylabel('Accuracy', fontsize=10)
        plt.xlabel('Epoch', fontsize=10)
        plt.legend(['Train', 'Test'], loc='upper left')


        plt.subplot(122)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss', fontsize=10)
        plt.ylabel('Loss', fontsize=10)
        plt.xlabel('Epoch', fontsize=10)
        plt.legend(['Train', 'Test'], loc='upper left')

        save = "cnn_1_"+optimizer+"_"+str(epoch)
        plt.savefig(save)
        plt.show()
vgg = VGG16()
vgg_layer_list = vgg.layers

model_vgg = Sequential()
for i in range(len(vgg_layer_list)-1):
    model_vgg.add(vgg_layer_list[i])

for layers in model_vgg.layers:
    layers.trainable = False    
model_vgg.add(Dense(numberOfClass, activation="softmax"))
print(model_vgg.summary())
from keras.utils import plot_model
plot_model(model_vgg)
for epoch in epochs:
    print("Epoch:", epoch)
    for optimizer in optimizers:
        model_vgg.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
        
        
        print("Optimizer:",optimizer)
        history = model_vgg.fit_generator(
            generator=train_data,
            epochs=epoch,
            validation_data=test_data,
            steps_per_epoch=40,
            validation_steps=20)
        
        
        val_loss, val_acc = model_vgg.evaluate(test_data)
        print(val_loss)
        print(val_acc)
        model_vgg.save('vgg_0_num_reader.model')
        
        
        plt.figure(figsize=(30,5))
        plt.subplot(121)
        title = 'Model:VGG16 Epoch:'+str(epoch)+' Optimizer:'+optimizer+' Data Replication: False'
        plt.suptitle(title, fontsize=15)


        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy', fontsize=10)
        plt.ylabel('Accuracy', fontsize=10)
        plt.xlabel('Epoch', fontsize=10)
        plt.legend(['Train', 'Test'], loc='upper left')


        plt.subplot(122)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss', fontsize=10)
        plt.ylabel('Loss', fontsize=10)
        plt.xlabel('Epoch', fontsize=10)
        plt.legend(['Train', 'Test'], loc='upper left')


        save = "vgg_0_"+optimizer+"_"+str(epoch)
        plt.savefig(save)
        plt.show()       
for epoch in epochs:
    print("Epoch:", epoch)
    for optimizer in optimizers:
        model_vgg.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
        
             
        print("Optimizer:",optimizer)
        history = model_vgg.fit_generator(
            generator=train_generator,
            epochs=epoch,
            validation_data=test_generator,
            steps_per_epoch=40,
            validation_steps=20)
        
        val_loss, val_acc = model_vgg.evaluate(test_data)
        print(val_loss)
        print(val_acc)
        model_vgg.save('vgg_1_num_reader.model')
        
        plt.figure(figsize=(30,5))
        plt.subplot(121)
        title = 'Model:VGG16 Epoch:'+str(epoch)+' Optimizer:'+optimizer+' Data Replication: True'
        plt.suptitle(title, fontsize=15)


        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy', fontsize=10)
        plt.ylabel('Accuracy', fontsize=10)
        plt.xlabel('Epoch', fontsize=10)
        plt.legend(['Train', 'Test'], loc='upper left')


        plt.subplot(122)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss', fontsize=10)
        plt.ylabel('Loss', fontsize=10)
        plt.xlabel('Epoch', fontsize=10)
        plt.legend(['Train', 'Test'], loc='upper left')

        save = "vgg_1_"+optimizer+"_"+str(epoch)
        plt.savefig(save)
        plt.show()
model_resnet = Sequential()
model_resnet.add(ResNet50(include_top=False, weights='imagenet', pooling = 'avg'))
model_resnet.add(Dense(numberOfClass, activation="softmax"))
model_resnet.layers[0].trainable = False
model_resnet.summary()
from keras.utils import plot_model
plot_model(model_resnet)
for epoch in epochs:
    print("Epoch:", epoch)
    for optimizer in optimizers:
        model_resnet.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
        
        
        print("Optimizer:",optimizer)
        history = model_resnet.fit_generator(
            generator=train_data,
            epochs=epoch,
            validation_data=test_data,
            steps_per_epoch=40,
            validation_steps=20)
        
        val_loss, val_acc = model_resnet.evaluate(test_data)
        print(val_loss)
        print(val_acc)
        model_resnet.save('resnet_0_num_reader.model')
        
        plt.figure(figsize=(30,5))
        plt.subplot(121)
        title = 'Model:ResNet50 Epoch:'+str(epoch)+' Optimizer:'+optimizer+' Data Replication: False'
        plt.suptitle(title, fontsize=15)


        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy', fontsize=10)
        plt.ylabel('Accuracy', fontsize=10)
        plt.xlabel('Epoch', fontsize=10)
        plt.legend(['Train', 'Test'], loc='upper left')


        plt.subplot(122)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss', fontsize=10)
        plt.ylabel('Loss', fontsize=10)
        plt.xlabel('Epoch', fontsize=10)
        plt.legend(['Train', 'Test'], loc='upper left')


        save = "resnet_0_"+optimizer+"_"+str(epoch)
        plt.savefig(save)
        plt.show()
for epoch in epochs:
    print("Epoch:", epoch)
    for optimizer in optimizers:
        model_resnet.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
        
        print("Optimizer:",optimizer)
        history = model_resnet.fit_generator(
            generator=train_generator,
            epochs=epoch,
            validation_data=test_generator,
            steps_per_epoch=40,
            validation_steps=20)
        
        val_loss, val_acc = model_resnet.evaluate(test_data)
        print(val_loss)
        print(val_acc)
        model_resnet.save('resnet_1_num_reader.model')
        
        plt.figure(figsize=(30,5))
        plt.subplot(121)
        title = 'Model:ResNet50 Epoch:'+str(epoch)+' Optimizer:'+optimizer+' Data Replication: True'
        plt.suptitle(title, fontsize=15)


        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy', fontsize=10)
        plt.ylabel('Accuracy', fontsize=10)
        plt.xlabel('Epoch', fontsize=10)
        plt.legend(['Train', 'Test'], loc='upper left')


        plt.subplot(122)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss', fontsize=10)
        plt.ylabel('Loss', fontsize=10)
        plt.xlabel('Epoch', fontsize=10)
        plt.legend(['Train', 'Test'], loc='upper left')


        save = "resnet_1_"+optimizer+"_"+str(epoch)
        plt.savefig(save)
        plt.show()
model.compile(loss='sparse_categorical_crossentropy',
      optimizer='Adam',
      metrics=['accuracy'])

history = model.fit_generator(
    generator=train_generator,
    epochs=5,
    validation_data=test_generator,
    steps_per_epoch=40,
    validation_steps=20)
plt.figure(figsize = (35,5))
plt.suptitle('Real signatures Test', fontsize=20)
x, y = 1, 10
for i in range(10):
    plt.subplot(x, y, i+1)
    plt.axis('off')
    img = cv2.resize(cv2.imread(test_g[i], 1), (224,224))
    plt.imshow(img)
    img = img_to_array(img)
    img = img.reshape(1,224,224,3)
    pre = model.predict_classes(img, batch_size=1)
    plt.title(pre, fontsize=20)
plt.savefig('test_g_cnn')
plt.show()


plt.figure(figsize = (35,5))
plt.suptitle('Fake signatures Test', fontsize=20)
x, y = 1, 10
for i in range(10):
    plt.subplot(x, y, i+1)
    plt.axis('off')
    img = cv2.resize(cv2.imread(test_f[i], 1), (224,224))
    plt.imshow(img)
    img = img_to_array(img)
    img = img.reshape(1,224,224,3)
    pre = model.predict_classes(img, batch_size=1)
    plt.title(pre, fontsize=20)
plt.savefig('test_f_cnn')
plt.show()
model_vgg.compile(loss='sparse_categorical_crossentropy',
      optimizer='RMSprop',
      metrics=['accuracy'])

history = model_vgg.fit_generator(
    generator=train_generator,
    epochs=10,
    validation_data=test_generator,
    steps_per_epoch=40,
    validation_steps=20)
plt.figure(figsize = (35,5))
plt.suptitle('Real signatures Test', fontsize=20)
x, y = 1, 55
for i in range(55):
    plt.subplot(x, y, i+1)
    plt.axis('off')
    img = cv2.resize(cv2.imread(test_g[i], 1), (224,224))
    plt.imshow(img)
    img = img_to_array(img)
    img = img.reshape(1,224,224,3)
    pre = model_vgg.predict_classes(img, batch_size=1)
    plt.title(pre, fontsize=20)
plt.savefig('test_g_vgg')
plt.show()


plt.figure(figsize = (35,5))
plt.suptitle('Fake signatures Test', fontsize=20)
x, y = 1, 55
for i in range(55):
    plt.subplot(x, y, i+1)
    plt.axis('off')
    img = cv2.resize(cv2.imread(test_f[i], 1), (224,224))
    plt.imshow(img)
    img = img_to_array(img)
    img = img.reshape(1,224,224,3)
    pre = model_vgg.predict_classes(img, batch_size=1)
    plt.title(pre, fontsize=20)
plt.savefig('test_f_vgg')
plt.show()
model_resnet.compile(loss='sparse_categorical_crossentropy',
      optimizer='RMSprop',
      metrics=['accuracy'])

history = model_resnet.fit_generator(
    generator=train_generator,
    epochs=15,
    validation_data=test_generator,
    steps_per_epoch=40,
    validation_steps=20)
plt.figure(figsize = (35,5))
plt.suptitle('Real signatures Test', fontsize=20)
x, y = 1, 10
for i in range(10):
    plt.subplot(x, y, i+1)
    plt.axis('off')
    img = cv2.resize(cv2.imread(test_g[i], 1), (224,224))
    plt.imshow(img)
    img = img_to_array(img)
    img = img.reshape(1,224,224,3)
    pre = model_resnet.predict_classes(img, batch_size=1)
    plt.title(pre, fontsize=20)
plt.savefig('test_g_resnet')
plt.show()


plt.figure(figsize = (35,5))
plt.suptitle('Fake signatures Test', fontsize=20)
x, y = 1, 10
for i in range(10):
    plt.subplot(x, y, i+1)
    plt.axis('off')
    img = cv2.resize(cv2.imread(test_f[i], 1), (224,224))
    plt.imshow(img)
    img = img_to_array(img)
    img = img.reshape(1,224,224,3)
    pre = model_resnet.predict_classes(img, batch_size=1)
    plt.title(pre, fontsize=20)
plt.savefig('test_f_resnet')
plt.show()