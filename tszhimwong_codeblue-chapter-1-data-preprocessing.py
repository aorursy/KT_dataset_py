# import system modules

import os

import sys

import datetime

import random



# import external helpful libraries

import tensorflow as tf

import numpy as np

import cv2

import h5py

import matplotlib.pyplot as plt

from tqdm import tqdm

import pandas as pd

import imgaug as ia

import imgaug.augmenters as iaa



# import keras

import keras

from keras import backend as K

from keras.models import Sequential, Model

from keras.layers import Dense, Dropout, Activation, Flatten, Input, Lambda, Reshape

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization 

from keras.layers import Input, UpSampling2D, concatenate  

from keras.optimizers import Nadam, SGD

from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, TensorBoard



# possible libraries for metrics

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score



#K-Fold Cross Validation

import sklearn

from sklearn.model_selection import train_test_split



# Set the random seed to ensure reproducibility

np.random.seed(1234)

tf.random.set_seed(1234)
# Edit this variable to point to your dataset, if needed

dataset_path = '../input/gametei2020/dataset'



#Form a full dataset 

def combine_dataset(dataset_path, split1, split2):

    split1_path = os.path.join(dataset_path, split1)

    split2_path = os.path.join(dataset_path, split2)

    data_out = []

    

    # iterate each class

    classes = ["NORMAL", "PNEUMONIA"]

    # notice that class_idx = 0 for NORMAL, 1 for PNEUMONIA

    for class_idx, _class in enumerate(classes):

        class_path1 = os.path.join(split1_path, _class) # path to each class dir

        class_path2 = os.path.join(split2_path, _class)

        # iterate through all files in dir

        for filename in os.listdir(class_path1):

            # ensure files are images, if so append to output

            if filename.endswith(".jpeg"):

                img_path = os.path.join(class_path1, filename)

                data_out.append((img_path, class_idx))

        for filename in os.listdir(class_path2):

            # ensure files are images, if so append to output

            if filename.endswith(".jpeg"):

                img_path = os.path.join(class_path2, filename)

                data_out.append((img_path, class_idx))

                

    return data_out

dataset_seq = combine_dataset(dataset_path,split1 = "train",split2 = "val")

dataset_pneumonia_cases = sum([class_idx for (img_path, class_idx) in dataset_seq])

dataset_normal_cases = len(dataset_seq) - dataset_pneumonia_cases

print("Combined - Total: %d, Normal: %d, Pneumonia: %d" % (len(dataset_seq), dataset_normal_cases, dataset_pneumonia_cases))
# Shuffle the dataset

from sklearn.utils import shuffle

dataset_seq = shuffle(dataset_seq)



# Kfold splitting 

n_folds = 4

fold_seq = [[] for i in range(n_folds)]

fold_seq[0] = dataset_seq[0:1308]

fold_seq[1] = dataset_seq[1308:2616]

fold_seq[2] = dataset_seq[2616:3924]

fold_seq[3] = dataset_seq[3924:5232]

fold_pneumonia_cases = [[] for i in range(n_folds)]

fold_normal_cases = [[] for i in range(n_folds)]

for i in range(n_folds):

    fold_pneumonia_cases[i] = sum([class_idx for (img_path, class_idx) in fold_seq[i]])

    fold_normal_cases[i] = len(fold_seq[i]) - fold_pneumonia_cases[i]

    print("Fold %d - Total: %d, Normal: %d, Pneumonia: %d" % (i, len(fold_seq[i]), fold_normal_cases[i], fold_pneumonia_cases[i]))
#Dividing training sets intor normal and pneumonia



n_fold_pneumonia_cases = []

n_fold_normal_cases = []

fold_normal = [[] for i in range(4)]

fold_pneumonia = [[] for i in range(4)]



for j in range(n_folds):

    fold_normal[j] = [[] for i in range(len(fold_seq[i]))]

    fold_pneumonia[j] = [[] for i in range(len(fold_seq[i]))]

    fold_normal[j] = [[i[0],i[1]] for i in fold_seq[j] if i[1]== 0]

    fold_pneumonia[j] = [[i[0],i[1]] for i in fold_seq[j] if i[1]== 1]

    #print(len(fold_normal[j]))

    #print(len(fold_pneumonia[j]))    

    for k in range(len(fold_pneumonia[j]) - len(fold_normal[j])):

        sample = random.sample(fold_normal[j],1)

        fold_seq[j] = fold_seq[j] + sample

    #print(len(fold_seq[j]))

    n_fold_pneumonia_cases.append (sum([class_idx for (img_path, class_idx) in fold_seq[j]])) #compute pneumonia cases by summing the total number of 1's

    n_fold_normal_cases.append (len(fold_seq[j]) - n_fold_pneumonia_cases [j])                       # subtract from total to get normal cases

    print("Oversampled Train split %d - Total: %d, Normal: %d, Pneumonia: %d" % (j+1, len(fold_seq[j]), n_fold_normal_cases[j], n_fold_pneumonia_cases[j]))
#Saving The splits

split_df = pd.DataFrame(columns=['img', 'class', 'fold'])

count = 0

for i in range(n_folds):

    fold_num = i+1

    for j in range(len(fold_seq[i])):

        img_idx = fold_seq[i][j][0]

        class_idx =  fold_seq[i][j][1]

        split_df.loc[count] = [img_idx] + [class_idx] + [fold_num]

        count +=1

split_df.to_csv("Split.csv", index=False)
# Create a matplotlib plot of the images

fig, ax = plt.subplots(2, 5, figsize=(30,10))

for x in range(2):

    # set the boundaries for getting idx for NORMAL

    # since the images are loaded such that normal images

    # are loaded first, it always starts from index 0.

    # the ending index is computed based on the number of

    # training normal images

    if x == 0:

        low = 0

        high = dataset_normal_cases # high is exclusive

    # get the boundary indices for PNEUMONIA

    else:

        low = dataset_normal_cases

        high = len(dataset_seq)

    for y in range(5):

        # select a random image

        sample_idx = np.random.randint(low, high)

        # read the image

        sample = cv2.imread(dataset_seq[sample_idx][0])

        # display image

        ax[x, y].imshow(sample, cmap='gray')

        # set title of plots

        ax[x, y].set_title("Normal" if x == 0 else "Pneumonia")

plt.show()
