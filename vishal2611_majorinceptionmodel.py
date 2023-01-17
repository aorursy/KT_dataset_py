# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import os

import matplotlib.gridspec as gridspec

import seaborn as sns

import zlib

import itertools

import sklearn

import itertools

import scipy

import skimage

from skimage.transform import resize

import csv

from tqdm import tqdm

from sklearn import model_selection

from sklearn.model_selection import train_test_split, learning_curve,KFold,cross_val_score,StratifiedKFold

from sklearn.utils import class_weight

from sklearn.metrics import confusion_matrix

import keras

from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Lambda, MaxPool2D, BatchNormalization

from keras.utils import np_utils

from keras.utils.np_utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

from keras import models, layers, optimizers

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.utils import class_weight

from keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta, RMSprop

from keras.models import Sequential, model_from_json

from keras.layers import Activation,Dense, Dropout, Flatten, Conv2D, MaxPool2D,MaxPooling2D,AveragePooling2D, BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from keras import backend as K

from keras.applications.vgg16 import VGG16

from keras.applications.inception_v3 import InceptionV3

from keras.models import Model

from keras.applications.inception_v3 import InceptionV3

from imblearn.over_sampling import RandomOverSampler

from imblearn.under_sampling import RandomUnderSampler

from keras.layers import Input

from keras.models import Model

import matplotlib.pyplot as plt

from keras.applications.densenet import DenseNet121, preprocess_input

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Input

from keras.models import Model

from keras.layers import Dense

from keras.optimizers import Adam

#from generator import DataGenerator

import keras

from keras.callbacks import ModelCheckpoint

%matplotlib inline

import cv2

import warnings

warnings.filterwarnings("ignore")
labels = os.listdir('../input/oct2017/OCT2017 /train/')

train_datagen = ImageDataGenerator(samplewise_center=True, 

                              samplewise_std_normalization=True, 

                              horizontal_flip = True, 

                              vertical_flip = False, 

                              height_shift_range= 0.05, 

                              width_shift_range=0.1, 

                              rotation_range=15, 

                              zoom_range=0.15,

                                   validation_split=0.2)
IMG_SIZE = 224

batch_size = 32

train_data_dir = '../input/oct2017/OCT2017 /train'

validation_data_dir = '../input/oct2017/OCT2017 /val'

train_generator = train_datagen.flow_from_directory(

    train_data_dir,

    target_size=(IMG_SIZE , IMG_SIZE),

    batch_size=16,

    subset='training',

    class_mode='categorical')

valid_X, valid_Y = next(train_datagen.flow_from_directory(

    train_data_dir,

    target_size=(IMG_SIZE , IMG_SIZE),

    batch_size=4000,

    subset='validation',

    class_mode='categorical'))
t_x, t_y = next(train_generator)

fig, m_axs = plt.subplots(4, 4, figsize = (16, 16))

for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):

    c_ax.imshow(c_x[:,:,0], cmap = 'bone')

    c_ax.set_title(', '.join([n_class for n_class, n_score in zip(labels, c_y) 

                             if n_score>0.5]))

    c_ax.axis('off')
def inception_v3():

    img_in = Input(t_x.shape[1:])              #input of model 

    model =  InceptionV3(include_top= False , # remove  the 3 fully-connected layers at the top of the network

                weights='imagenet',      # pre train weight 

                input_tensor= img_in, 

                input_shape= t_x.shape[1:],

                pooling ='avg') 

    x = model.output  

    predictions = Dense(4, activation="softmax", name="predictions")(x)    # fuly connected layer for predict class 

    model = Model(inputs=img_in, outputs=predictions)

    return model
model = inception_v3()

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',

                           metrics = ['accuracy'])
from keras.callbacks import ModelCheckpoint

filepath="weights.best.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

callbacks_list = [checkpoint]
history = model.fit_generator(train_generator, 

                                  steps_per_epoch=100,

                                  validation_data = (valid_X,valid_Y), 

                                  epochs = 30,

                                  callbacks=callbacks_list)
test_X, test_Y = next(train_datagen.flow_from_directory(

    train_data_dir,

    target_size=(IMG_SIZE , IMG_SIZE),

    batch_size=8000,

    subset='validation',

    class_mode='categorical'))
model = inception_v3()

model.load_weights(filepath)
y_pred = model.predict(test_X)
pred_class = []

for i in range(len(y_pred)):

    pred_class.append(np.argmax(y_pred[i]))
actual_class = []

for i in range(len(test_Y)):

    actual_class.append(np.argmax(test_Y[i]))
print('accuracy = ',accuracy_score(pred_class,actual_class))
vishal= inception_v3()

vishal.load_weights(filepath)
y_pred1= vishal.predict(test_X)
pred_class1 = []

for i in range(len(y_pred1)):

    pred_class1.append(np.argmax(y_pred1[i]))



    

print(pred_class1)
print('accuracy = ',accuracy_score(pred_class1,actual_class))
def plot_confusion_matrix(test_y, predict_y):

    C = confusion_matrix(test_y, predict_y)

    # C = 9,9 matrix, each cell (i,j) represents number of points of class i are predicted class j

    

    A =(((C.T)/(C.sum(axis=1))).T)

    #divid each element of the confusion matrix with the sum of elements in that column

    

    # C = [[1, 2],

    #     [3, 4]]

    # C.T = [[1, 3],

    #        [2, 4]]

    # C.sum(axis = 1)  axis=0 corresonds to columns and axis=1 corresponds to rows in two diamensional array

    # C.sum(axix =1) = [[3, 7]]

    # ((C.T)/(C.sum(axis=1))) = [[1/3, 3/7]

    #                           [2/3, 4/7]]



    # ((C.T)/(C.sum(axis=1))).T = [[1/3, 2/3]

    #                           [3/7, 4/7]]

    # sum of row elements = 1

    

    B =(C/C.sum(axis=0))

    #divid each element of the confusion matrix with the sum of elements in that row

    # C = [[1, 2],

    #     [3, 4]]

    # C.sum(axis = 0)  axis=0 corresonds to columns and axis=1 corresponds to rows in two diamensional array

    # C.sum(axix =0) = [[4, 6]]

    # (C/C.sum(axis=0)) = [[1/4, 2/6],

    #                      [3/4, 4/6]] 

    

    labels = [0,1,2,3]

    # representing A in heatmap format

    print("-"*20, "Confusion matrix", "-"*20)

    plt.figure(figsize=(20,7))

    sns.heatmap(C, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)

    plt.xlabel('Predicted Class')

    plt.ylabel('Original Class')

    plt.show()



    print("-"*20, "Precision matrix (Columm Sum=1)", "-"*20)

    plt.figure(figsize=(20,7))

    sns.heatmap(B, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)

    plt.xlabel('Predicted Class')

    plt.ylabel('Original Class')

    plt.show()

    

    # representing B in heatmap format

    print("-"*20, "Recall matrix (Row sum=1)", "-"*20)

    plt.figure(figsize=(20,7))

    sns.heatmap(A, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)

    plt.xlabel('Predicted Class')

    plt.ylabel('Original Class')

    plt.show()
plot_confusion_matrix(actual_class,pred_class)