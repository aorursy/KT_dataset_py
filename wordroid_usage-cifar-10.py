import os
print(os.listdir("../input"))
import os.path
import itertools
from itertools import chain

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn import cluster, datasets, mixture
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

from keras.layers import Input, Embedding, LSTM, GRU, Dense, Dropout, Lambda, \
    Conv1D, Conv2D, Conv3D, \
    Conv2DTranspose, \
    AveragePooling1D, AveragePooling2D, \
    MaxPooling1D, MaxPooling2D, MaxPooling3D, \
    GlobalAveragePooling1D, \
    GlobalMaxPooling1D, GlobalMaxPooling2D, GlobalMaxPooling3D, \
    LocallyConnected1D, LocallyConnected2D, \
    concatenate, Flatten, Average, Activation, \
    RepeatVector, Permute, Reshape, Dot, \
    multiply, dot, add, \
    PReLU, \
    Bidirectional, TimeDistributed, \
    SpatialDropout1D, \
    BatchNormalization
from keras.models import Model, Sequential
from keras import losses
from keras.callbacks import BaseLogger, ProgbarLogger, Callback, History
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.wrappers.scikit_learn import KerasClassifier
from keras import regularizers
from keras import initializers
from keras.metrics import categorical_accuracy
from keras.constraints import maxnorm, non_neg
from keras.optimizers import RMSprop
from keras.utils import to_categorical, plot_model
from keras import backend as K
from PIL import Image
from zipfile import ZipFile
import h5py
import cv2
from tqdm import tqdm
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
batch1 = unpickle("../input/data_batch_1")
batch2 = unpickle("../input/data_batch_2")
batch3 = unpickle("../input/data_batch_3")
batch4 = unpickle("../input/data_batch_4")
batch5 = unpickle("../input/data_batch_5")
test_batch = unpickle("../input/test_batch")
def load_data0(btch):
    labels = btch[b'labels']
    imgs = btch[b'data'].reshape((-1, 32, 32, 3))
    
    res = []
    for ii in range(imgs.shape[0]):
        img = imgs[ii].copy()
        #img = np.transpose(img.flatten().reshape(3,32,32))
        img = np.fliplr(np.rot90(np.transpose(img.flatten().reshape(3,32,32)), k=-1))
        res.append(img)
    imgs = np.stack(res)
    return labels, imgs

labels, imgs = load_data0(batch1)
imgs.shape

def load_data():
    x_train_l = []
    y_train_l = []
    for ibatch in [batch1, batch2, batch3, batch4, batch5]:
        labels, imgs = load_data0(ibatch)
        x_train_l.append(imgs)
        y_train_l.extend(labels)
    x_train = np.vstack(x_train_l)
    y_train = np.vstack(y_train_l)
    
    x_test_l = []
    y_test_l = []
    labels, imgs = load_data0(test_batch)
    x_test_l.append(imgs)
    y_test_l.extend(labels)
    x_test = np.vstack(x_test_l)
    y_test = np.vstack(y_test_l)
    
    
    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_data()

del batch1, batch2, batch3, batch4, batch5, test_batch
y_train
y_test
x_train.shape
x_test.shape
plt.imshow(x_train[0])
plt.imshow(x_test[0])

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train_fl = x_train.reshape((x_train.shape[0], -1))
print(x_train_fl.shape)
x_test_fl = x_test.reshape((x_test.shape[0], -1))
print(x_test_fl.shape)
y_train = y_train.flatten()
y_train
y_cat_train = to_categorical(y_train)
print(y_cat_train.shape)

y_test = y_test.flatten()
y_test

y_cat_test = to_categorical(y_test)
print(y_cat_test.shape)
plt.imshow(x_train[0])
nrows=10
ncols=10
fig, subs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 10))

for ii in range(nrows):
    for jj in range(ncols):
        iplt = subs[ii, jj]
        img_array = x_train[ii*ncols + jj]
        iplt.imshow(img_array)
nrows=10
ncols=12
fig, subs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 10))

for ii in range(nrows):
    idx = (y_train == ii)
    target_img = x_train[idx][:ncols]
    for jj in range(ncols):
        iplt = subs[ii, jj]
        img_array = target_img[jj]
        iplt.imshow(img_array)
nrows=10
ncols=12
fig, subs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 10))

for ii in range(nrows):
    idx = (y_train == ii)
    target_img = x_train_fl[idx][:ncols]
    for jj in range(ncols):
        iplt = subs[ii, jj]
        img_array = target_img[jj].reshape((32,32,3))
        iplt.imshow(img_array)
