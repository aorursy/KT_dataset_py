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
#Creating Data Generator

class xray_data_generator(keras.utils.Sequence):

    """

    Data generator derived from Keras' Sequence to be used with fit_generator.

    """

    def __init__(self, seq, dims=(331,331), batch_size=32, shuffle=True):

        # Save params into self

        self.dims = dims

        self.batch_size = batch_size

        self.seq = seq

        self.shuffle = shuffle

        

        # create data augmentor

        self.aug = iaa.SomeOf((0,3),[

                #iaa.Fliplr(), # horizontal flips

                iaa.Affine(scale={"x": (0.75, 1.25), "y": (0.75, 1.25)}),

                iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}),    

                iaa.Affine(rotate=(-10, 10)), # rotate images

                iaa.Multiply((0.8, 1.2)),  #random brightness

                iaa.Affine(shear=(-10, 10)),

                #iaa.GammaContrast((0.8, 1.2)),

                iaa.GaussianBlur(sigma=(0.0, 1.0))

                                    ],

                random_order=True

                             )



        # shuffle the dataset

        if self.shuffle:

          random.shuffle(self.seq)    



    def get_data(self, index):

        '''

        Given an index, retrieve the image and apply processing,

        including resizing and converting color encoding. This is

        where data augmentation can be added if desired.

        '''

        img_path, class_idx = self.seq[index]

        # Load the image

        img = cv2.imread(img_path)

        img = cv2.resize(img, self.dims)



        # if grayscale, convert to RGB

        if img.shape[-1] == 1:

            img = np.stack((img,img,img), axis=-1)



        # by default, cv2 reads images in using BGR format

        # we want to convert it to RGB

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



        # normalize values to [0, 1]

        img = img.astype(np.float32)/255.



        # augment image

        img = self.aug.augment_image(img)



        # Load the labels

        label = keras.utils.to_categorical(class_idx, num_classes=2)

        

        return img, label

      

    def get_classes(self):

        class_idxs = [class_idx for _, class_idx in self.seq]

        return np.array(class_idxs)



    def __len__(self):

        '''

        Returns the number of batches per epoch.

        Used by Keras' fit_generator to determine number of training steps.

        '''

        return int(np.floor(len(self.seq) / self.batch_size))



    def __getitem__(self, index):

        '''

        Actual retrieval of batch data during training.

        Data is retrieved by calling self.get_data on an index

        which is then batched, and returned

        '''

        # create empty batches

        batch_img = np.empty((self.batch_size,) + self.dims + (3,))

        batch_label = np.empty((self.batch_size,) + (2,))



        # load the images and labels into the batch

        # using the get_data method defined above

        for i in range(self.batch_size):

            img, label = self.get_data(index*self.batch_size+i)    

            batch_img[i] = img

            batch_label[i] = label



        return batch_img, batch_label



    def on_epoch_end(self):

        '''

        Shuffles the data sequence after each epoch

        '''

        if self.shuffle:

          random.shuffle(self.seq)
# Edit this variable to point to your dataset, if needed

dataset_path = "../input/gametei2020/dataset"



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
#Loading Splits

split_df = pd.read_csv("../input/splitting/Split.csv")

n_folds = 4

fold_seq = [[]for i in range(n_folds)]



for i in range(dataset_pneumonia_cases*2):

    if split_df['fold'][i] == 1:

        fold_seq[0].append((split_df['img'][i],split_df['class'][i]))

    if split_df['fold'][i] == 2:

        fold_seq[1].append((split_df['img'][i],split_df['class'][i]))

    if split_df['fold'][i] == 3:

        fold_seq[2].append((split_df['img'][i],split_df['class'][i]))

    if split_df['fold'][i] == 4:

        fold_seq[3].append((split_df['img'][i],split_df['class'][i]))   



n_fold_pneumonia_cases = []

n_fold_normal_cases = []



for j in range(n_folds):   

    n_fold_pneumonia_cases.append (sum([class_idx for (img_path, class_idx) in fold_seq[j]])) #compute pneumonia cases by summing the total number of 1's

    n_fold_normal_cases.append (len(fold_seq[j]) - n_fold_pneumonia_cases [j])                       # subtract from total to get normal cases

    print("Oversampled Train split %d - Total: %d, Normal: %d, Pneumonia: %d" % (j+1, len(fold_seq[j]), n_fold_normal_cases[j], n_fold_pneumonia_cases[j]))
#DenseNet201

from keras import layers

from keras import models

from keras import optimizers

from keras.applications import DenseNet201

from keras.layers import Dense, GlobalAveragePooling2D

from keras.preprocessing.image import img_to_array, load_img

from keras.models import Model

from keras import backend as K



#Define the Base Model

input_shape=(331,331,3)



base_model = DenseNet201(

    include_top=False,

    weights='imagenet',

    input_shape=input_shape,

)

base_model.trainable = True



#Adding dense layers into the pretrained DenseNet201 model

model = models.Sequential()

model.add(base_model)

model.add(layers.GlobalAveragePooling2D())

#model.add(BatchNormalization())

#model.add(layers.Dense(1024, activation='elu',kernel_initializer='he_uniform', kernel_regularizer='l2'))

#model.add(layers.Dropout(0.5))

#model.add(layers.Dense(512, activation='elu',kernel_initializer='he_uniform', kernel_regularizer='l2'))

#model.add(layers.Dropout(0.5))

model.add(layers.Dense(256, activation='elu',kernel_initializer='he_uniform', kernel_regularizer='l2'))

model.add(layers.Dropout(0.5))

#model.add(layers.Dense(128, activation='elu',kernel_initializer='he_uniform', kernel_regularizer='l2'))

#model.add(layers.Dropout(0.5))

model.add(layers.Dense(2, activation='sigmoid'))



#model = Model(inputs=base_model.input, outputs=x)

model.summary()

#for i in range(14):

# model.layers[i+1].trainable = False
# Setup training parameters

learning_rate = 1e-4

epochs = 40

early_stop_patience = 5



# Define optimizer

# Here we are using the Adam optimizer which is usually a good starting point

optimizer = Nadam(lr=learning_rate)

# optimizer = SGD(lr=learning_rate)   # SGD Optimizer



# Define callbacks for training

# Early stop allows us to stop the training when there is no perceived

# improvement anymore, based on the defined patience

# COMMENTED OUT FOR BASELINE MODEL, PROVIDED FOR YOUR EASE OF USE

early_stop = EarlyStopping(monitor='val_loss',

                            patience=early_stop_patience, 

                            verbose=1, 

                            mode='min')





# Set up tensorboard for logging

# setup logdir based on datetime

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")   # rename the folders for each trial as you want

tensorboard_callback = TensorBoard(log_dir=log_dir)



#Using focal loss as loss function 

def focal_loss(alpha=0.25,gamma=2.0):

    def focal_crossentropy(y_true, y_pred):

        bce = K.binary_crossentropy(y_true, y_pred)

        

        y_pred = K.clip(y_pred, K.epsilon(), 1.- K.epsilon())

        p_t = (y_true*y_pred) + ((1-y_true)*(1-y_pred))

        

        alpha_factor = 1

        modulating_factor = 1



        alpha_factor = y_true*alpha + ((1-alpha)*(1-y_true))

        modulating_factor = K.pow((1-p_t), gamma)



        # compute the final loss and return

        return K.mean(alpha_factor*modulating_factor*bce, axis=-1)

    return focal_crossentropy



# Compile the model based on our defined metrics and optimizer        

model.compile(loss=focal_loss(), 

            metrics=['AUC'], 

            optimizer=optimizer) 
val_seq = []

train_seq = []

val_seq = fold_seq[3]  #We choose fold 4 as the validation data

for j in range(n_folds): #Other folds are used for training

    if (j!=3): 

        train_seq += fold_seq[j] 



train_pneumonia_cases = sum([class_idx for (img_path, class_idx) in train_seq])   # compute pneumonia cases by summing the total number of 1's

train_normal_cases = len(train_seq) - train_pneumonia_cases                       # subtract from total to get normal cases

val_pneumonia_cases = sum([class_idx for (img_path, class_idx) in val_seq])       # compute pneumonia cases for validation dataset

val_normal_cases = len(val_seq) - val_pneumonia_cases                             # compute normal cases for validation dataset



print("Train - Total: %d, Normal: %d, Pneumonia: %d" % (len(train_seq), train_normal_cases, train_pneumonia_cases))

print("Validation - Total: %d, Normal: %d, Pneumonia: %d" % (len(val_seq), val_normal_cases, val_pneumonia_cases))

train_gen = xray_data_generator(train_seq) #input into data_generator

val_gen = xray_data_generator(val_seq)     



#Introduce class weights

weight_for_0 = train_pneumonia_cases / len(train_seq)

weight_for_1 = train_normal_cases / len(train_seq)

class_weight = {0: weight_for_0, 1: weight_for_1}

#print(class_weight)



#Batch Sizes and Step Sizes

batch_size = 32

training_step_size = len(train_seq)//batch_size

validation_step_size = len(val_seq)//batch_size



# ModelCheckpoint saves the best model weight so far.

checkpt = ModelCheckpoint(monitor='val_loss',

                        filepath = 'best_DenseNet201_final.h5',

                        mode = 'min',

                        save_best_only=True)
#Run the training

history = model.fit_generator(train_gen, 

                          epochs=epochs, 

                          verbose=1, 

                          validation_data=val_gen,

                          callbacks=[checkpt, 

                                    early_stop,   # COMMENTED OUT FOR BASELINE

                                    tensorboard_callback    # may cause warning abt on_train_batch_end, also may need to be commented out on GCP

                                    ],

                          class_weight=class_weight,

                          steps_per_epoch  = training_step_size,

                          validation_steps = validation_step_size

                          )

# Model evaluation 

model = keras.models.load_model('best_DenseNet201_final.h5', custom_objects={'focal_crossentropy': focal_loss})



# Create a validation generator that does not shuffle

# This will allow our predicted value to match our true values in sequence

noshuf_val_gen = xray_data_generator(val_seq, batch_size=2, shuffle=False)



# Predicted values

raw = np.array([])

raw = model.predict_generator(noshuf_val_gen)

preds = np.argmax(raw, axis=1)

# True values

trues = noshuf_val_gen.get_classes()



# Compute metrics

acc = accuracy_score(trues, preds)

prec = precision_score(trues, preds)

rec = recall_score(trues, preds)

f1 = f1_score(trues, preds)

auc = roc_auc_score(trues, preds)



# Print metrics summary

print("Evaluation of model on val split 4:")

print("Accuracy: %.3f" % acc)

print("Precision: %.3f" % prec)

print("Recall: %.3f" %  rec)

print("F1: %.3f" % f1)

print("AUC: %.3f" % auc)





from sklearn.metrics import confusion_matrix

results = confusion_matrix(trues, preds)

print(results)
from mlxtend.plotting import plot_confusion_matrix

plt.figure()

plot_confusion_matrix(results,figsize=(12,8), hide_ticks=True,cmap=plt.cm.Blues)

plt.xticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)

plt.yticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)

plt.show()