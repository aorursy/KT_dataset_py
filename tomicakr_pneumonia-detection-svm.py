import os

import numpy as np

import pandas as pd 

import random

import cv2

import matplotlib.pyplot as plt

import keras.backend as K

import tensorflow as tf

import warnings



from random import shuffle 

from tqdm import tqdm 

from PIL import Image

from keras.models import Model, Sequential

from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization, Conv2D, SeparableConv2D, MaxPool2D, LeakyReLU, Activation

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.preprocessing import MinMaxScaler

from sklearn import svm



%matplotlib inline



in_path = '../input/chest-xray-pneumonia//chest_xray/chest_xray/'
def extract_data(dimensions, batch_length):

    tgen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, vertical_flip=True)

    test_val_datagen = ImageDataGenerator(rescale=1./255)

    tgen_final = tgen.flow_from_directory(directory=in_path+'train', target_size=(dimensions, dimensions), batch_size=batch_length, class_mode='binary', shuffle=True)

    test_gen = test_val_datagen.flow_from_directory(directory=in_path+'test', target_size=(dimensions, dimensions), batch_size=batch_length, class_mode='binary', shuffle=True)



    test_data = []

    test_labels = []

    

    train_data = []

    train_labels = []



    for normal_image in (os.listdir(in_path + 'test' + '/NORMAL/')):

        normal_image = plt.imread(in_path+'test'+'/NORMAL/'+normal_image)

        normal_image = cv2.resize(normal_image, (dimensions, dimensions))

        normal_image = normal_image.astype('float32') / 255

        label = 0

        test_data.append(normal_image)

        test_labels.append(label)



    for pneumonia_image in (os.listdir(in_path + 'test' + '/PNEUMONIA/')):

        pneumonia_image = plt.imread(in_path+'test'+'/PNEUMONIA/'+pneumonia_image)

        pneumonia_image = cv2.resize(pneumonia_image, (dimensions, dimensions))

        pneumonia_image = pneumonia_image.astype('float32') / 255

        label = 1

        test_data.append(pneumonia_image)

        test_labels.append(label)



    for normal_image in (os.listdir(in_path + 'train' + '/NORMAL/')):

        if normal_image == '.DS_Store':

            continue

        normal_image = plt.imread(in_path+'train'+'/NORMAL/'+normal_image)

        normal_image = cv2.resize(normal_image, (dimensions, dimensions))

        normal_image = normal_image.astype('float32') / 255

        label = 0

        train_data.append(normal_image)

        train_labels.append(label)



    for pneumonia_image in (os.listdir(in_path + 'train' + '/PNEUMONIA/')):

        if pneumonia_image == '.DS_Store':

            continue

        pneumonia_image = plt.imread(in_path+'train'+'/PNEUMONIA/'+pneumonia_image)

        pneumonia_image = cv2.resize(pneumonia_image, (dimensions, dimensions))

        pneumonia_image = pneumonia_image.astype('float32') / 255

        label = 1

        train_data.append(pneumonia_image)

        train_labels.append(label)

    

    return tgen_final, test_gen, test_data, test_labels, train_data, train_labels
img_dims = 150

epochs = 3

batch_size = 10



train_gen, test_gen, test_data, test_labels, train_data, train_labels = extract_data(img_dims, batch_size)
flat_train_data = []

flat_test_data = []



for img in train_data: 

    flat_train_data.append(img.flatten())

    

for img in test_data: 

    flat_test_data.append(img.flatten())
listed_train_data = []

listed_test_data = []



for d in flat_train_data:

    listed_train_data.append(d.tolist()[:150*150])

    

for d in flat_test_data:

    listed_test_data.append(d.tolist()[:150*150])
train_labels_mini = train_labels

test_labels_mini = test_labels
clf = svm.SVC()

prediction = clf.fit(listed_train_data, train_labels_mini)
correctlyPredicted = 0

fp = 0

tp = 0

fn = 0

tn = 0



for ind in range(len(listed_test_data)): 

    pred = clf.predict([listed_test_data[ind]])

    real = test_labels_mini[ind]

    if real == 1 and pred == 1:

        tp += 1

    if real == 1 and pred == 0:

        fn += 1

    if real == 0 and pred == 1:

        fp += 1

    if real == 0 and pred == 0:

        tn += 1

    if pred == real:

        correctlyPredicted += 1



acc = correctlyPredicted/float(len(listed_test_data))

print("TEST DATA________________________")

print("Accuracy: {}".format(acc))

print("Confusion matrix: f_p = {}, t_p = {}, f_n = {}, t_n = {}".format(fp, tp, fn, tn))



prec = tp/(tp+fp)*100

rec = tp/(tp+fn)*100

print("Precision: {}, recall: {}".format(prec, rec))
correctlyPredicted = 0

fp = 0

tp = 0

fn = 0

tn = 0



for ind in range(len(listed_train_data)): 

    pred = clf.predict([listed_train_data[ind]])

    real = train_labels_mini[ind]

    if real == 1 and pred == 1:

        tp += 1

    if real == 1 and pred == 0:

        fn += 1

    if real == 0 and pred == 1:

        fp += 1

    if real == 0 and pred == 0:

        tn += 1

    if pred == real:

        correctlyPredicted += 1



acc = correctlyPredicted/float(len(listed_train_data))

print("TRAIN DATA________________________")

print("Accuracy: {}".format(acc))

print("Confusion matrix: f_p = {}, t_p = {}, f_n = {}, t_n = {}".format(fp, tp, fn, tn))



prec = tp/(tp+fp)*100

rec = tp/(tp+fn)*100

print("Precision: {}, recall: {}".format(prec, rec))