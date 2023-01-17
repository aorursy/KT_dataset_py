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

from keras.models import Model

from keras.applications.inception_v3 import InceptionV3

import os

from glob import glob

import matplotlib.pyplot as plt

import random

import cv2

import pandas as pd

import numpy as np

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

from imblearn.over_sampling import RandomOverSampler

from imblearn.under_sampling import RandomUnderSampler

#from keras.applications.mobilenet import MobileNet

#from sklearn.metrics import roc_auc_score

#from sklearn.metrics import roc_curve

#from sklearn.metrics import auc

#import warnings

#warnings.filterwarnings("ignore")

%matplotlib inline
# print(os.listdir("../input"))

# print(os.listdir("../input/asl-alphabet"))

# print(os.listdir("../input/asl-alphabet/asl_alphabet_train"))

# print(os.listdir("../input/asl-alphabet/asl_alphabet_train/asl_alphabet_train"))

# print(os.listdir("../input/asl-alphabet/asl_alphabet_train/asl_alphabet_train/A"))
imageSize=50

train_dir = "../input/asl-alphabet/asl_alphabet_train/asl_alphabet_train/"

test_dir =  "../input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/"

from tqdm import tqdm

def get_data(folder):

    """

    Load the data and labels from the given folder.

    """

    X = []

    y = []

    for folderName in os.listdir(folder):

        if not folderName.startswith('.'):

            if folderName in ['A']:

                label = 0

            elif folderName in ['B']:

                label = 1

            elif folderName in ['C']:

                label = 2

            elif folderName in ['D']:

                label = 3

            elif folderName in ['E']:

                label = 4

            elif folderName in ['F']:

                label = 5

            elif folderName in ['G']:

                label = 6

            elif folderName in ['H']:

                label = 7

            elif folderName in ['I']:

                label = 8

            elif folderName in ['J']:

                label = 9

            elif folderName in ['K']:

                label = 10

            elif folderName in ['L']:

                label = 11

            elif folderName in ['M']:

                label = 12

            elif folderName in ['N']:

                label = 13

            elif folderName in ['O']:

                label = 14

            elif folderName in ['P']:

                label = 15

            elif folderName in ['Q']:

                label = 16

            elif folderName in ['R']:

                label = 17

            elif folderName in ['S']:

                label = 18

            elif folderName in ['T']:

                label = 19

            elif folderName in ['U']:

                label = 20

            elif folderName in ['V']:

                label = 21

            elif folderName in ['W']:

                label = 22

            elif folderName in ['X']:

                label = 23

            elif folderName in ['Y']:

                label = 24

            elif folderName in ['Z']:

                label = 25

            elif folderName in ['del']:

                label = 26

            elif folderName in ['nothing']:

                label = 27

            elif folderName in ['space']:

                label = 28           

            else:

                label = 29

            for image_filename in tqdm(os.listdir(folder + folderName)):

                img_file = cv2.imread(folder + folderName + '/' + image_filename)

                if img_file is not None:

                    img_file = skimage.transform.resize(img_file, (imageSize, imageSize, 3))

                    img_arr = np.asarray(img_file)

                    X.append(img_arr)

                    y.append(label)

    X = np.asarray(X)

    y = np.asarray(y)

    return X,y

X_train, y_train = get_data(train_dir) 

#X_test, y_test= get_data(test_dir) # Too few images



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2) 



# Encode labels to hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])

from keras.utils.np_utils import to_categorical

y_trainHot = to_categorical(y_train, num_classes = 30)

y_testHot = to_categorical(y_test, num_classes = 30)
y_trainHot.shape
def plotHistogram(a):

    """

    Plot histogram of RGB Pixel Intensities

    """

    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)

    plt.imshow(a)

    plt.axis('off')

    histo = plt.subplot(1,2,2)

    histo.set_ylabel('Count')

    histo.set_xlabel('Pixel Intensity')

    n_bins = 30

    plt.hist(a[:,:,0].flatten(), bins= n_bins, lw = 0, color='r', alpha=0.5);

    plt.hist(a[:,:,1].flatten(), bins= n_bins, lw = 0, color='g', alpha=0.5);

    plt.hist(a[:,:,2].flatten(), bins= n_bins, lw = 0, color='b', alpha=0.5);

plotHistogram(X_train[1])
print("A")

multipleImages = glob('../input/asl-alphabet/asl_alphabet_train/asl_alphabet_train/A/**')

i_ = 0

plt.rcParams['figure.figsize'] = (10.0, 10.0)

plt.subplots_adjust(wspace=0, hspace=0)

for l in multipleImages[:25]:

    im = cv2.imread(l)

    im = cv2.resize(im, (128, 128)) 

    plt.subplot(5, 5, i_+1) #.set_title(l)

    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); plt.axis('off')

    i_ += 1
print("B")

multipleImages = glob('../input/asl-alphabet/asl_alphabet_train/asl_alphabet_train/B/**')

i_ = 0

plt.rcParams['figure.figsize'] = (10.0, 10.0)

plt.subplots_adjust(wspace=0, hspace=0)

for l in multipleImages[:25]:

    im = cv2.imread(l)

    im = cv2.resize(im, (128, 128)) 

    plt.subplot(5, 5, i_+1) #.set_title(l)

    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); plt.axis('off')

    i_ += 1
map_characters = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'del', 27: 'nothing', 28: 'space', 29: 'other'}

dict_characters=map_characters

import seaborn as sns

df = pd.DataFrame()

df["labels"]=y_train

lab = df['labels']

dist = lab.value_counts()

sns.countplot(lab)

print(dict_characters)
import tensorflow as tf



map_characters1 = map_characters

# Define the Model and train it normally



model = tf.keras.Sequential()

# Must define the input shape in the first layer of the neural network

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=5, padding='same', activation='relu', input_shape=(50, 50, 3))) 

model.add(tf.keras.layers.MaxPooling2D(pool_size=2))

model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))

model.add(tf.keras.layers.MaxPooling2D(pool_size=2))

model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'))

model.add(tf.keras.layers.MaxPooling2D(pool_size=2))

model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(256, activation='relu'))

model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(30, activation='softmax'))



model.summary()



model.compile(optimizer='adam',

              loss='categorical_crossentropy',

              metrics=['accuracy'])





model.fit(X_train, y_trainHot, epochs=12,batch_size=64)



#save model

model.save_weights("ASL_model.h5")





# To Do: (1) try using more than 30000 of the 87000 images; (2) try using larger images; (3) try using different network architectures 