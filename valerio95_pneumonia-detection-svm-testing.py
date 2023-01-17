import os

import numpy as np

import pandas as pd 

import random

import cv2

import matplotlib.pyplot as plt

import keras.backend as K

import tensorflow as tf



from keras.models import Model, Sequential

from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization, Conv2D, SeparableConv2D, MaxPool2D, LeakyReLU, Activation

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.svm import SVC

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler



%matplotlib inline



in_path = '../input/chest-xray-pneumonia//chest_xray/chest_xray/'
def extract_data(dimensions, batch_length):

    tgen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, vertical_flip=True)

    test_val_datagen = ImageDataGenerator(rescale=1./255)

    tgen_final = tgen.flow_from_directory(directory=in_path+'train', target_size=(dimensions, dimensions), batch_size=batch_length, class_mode='binary', shuffle=True)

    test_gen = test_val_datagen.flow_from_directory(directory=in_path+'test', target_size=(dimensions, dimensions), batch_size=batch_length, class_mode='binary', shuffle=True)



    test_data = []

    test_labels = []



    for nprmal_image in (os.listdir(in_path + 'test' + '/NORMAL/')):

        nprmal_image = plt.imread(in_path+'test'+'/NORMAL/'+nprmal_image)

        nprmal_image = cv2.resize(nprmal_image, (dimensions, dimensions))

        nprmal_image = np.dstack([nprmal_image, nprmal_image, nprmal_image])

        nprmal_image = nprmal_image.astype('float32') / 255

        label = 0

        test_data.append(nprmal_image)

        test_labels.append(label)



    for pneumonia_image in (os.listdir(in_path + 'test' + '/PNEUMONIA/')):

        pneumonia_image = plt.imread(in_path+'test'+'/PNEUMONIA/'+pneumonia_image)

        pneumonia_image = cv2.resize(pneumonia_image, (dimensions, dimensions))

        pneumonia_image = np.dstack([pneumonia_image, pneumonia_image, pneumonia_image])

        pneumonia_image = pneumonia_image.astype('float32') / 255

        label = 1

        test_data.append(pneumonia_image)

        test_labels.append(label)



    test_data = np.array(test_data)

    test_labels = np.array(test_labels)



    

    return tgen_final, test_gen, test_data, test_labels



#print (test_data.shape, test_labels.shape)
img_dims = 150

epochs = 10

batch_size = 32



tgen_final, test_gen, test_data, test_labels = extract_data(img_dims, batch_size)

ftest_data=test_data.flatten()

ftest_labels=test_labels.flatten()

features = np.hstack([ftest_data, ftest_labels])

global_features=features[0:6490*6490].reshape(-1,6490)



print(global_features.shape)

scaler = MinMaxScaler(feature_range=(0, 1))

rescaled_features = scaler.fit_transform(global_features)
from sklearn import model_selection





clf = models.append(('SVM', SVC(random_state=9)))

prediction= clf.fit(global_features.reshape(1,-1))[0]