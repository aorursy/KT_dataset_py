#Import necessary libraries which can be updated retroactively



import pandas as pd

import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt

%matplotlib inline

import keras

from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Input, Dense, LSTM, Embedding

from keras.layers import Dropout, Activation, Bidirectional, GlobalMaxPool1D

from keras.models import Model

from keras import initializers, regularizers, constraints, optimizers, layers, models

from keras.preprocessing import text, sequence

from keras.callbacks import ModelCheckpoint, EarlyStopping

import os, shutil

from glob import glob

import fileinput

from collections import defaultdict

import numpy as np

import os, shutil

from keras.utils import to_categorical

from keras.utils.vis_utils import plot_model

from keras.applications import VGG19, inception_v3

from sklearn.metrics import confusion_matrix

import itertools

import random as rn

import pydot

import graphviz
directory_list = ['train', 'test','val']

types = ['bacteria', 'virus']





#creates a new directory for each type of pneumonia

for directory in directory_list:

    current_directory = directory + '/'

    current_dirList = os.listdir(current_directory)

    

    for infection in types:

        new_dir = os.path.join(current_directory +'/'+infection)

        os.mkdir(new_dir)

        

        for jpeg in current_dirList:

            if infection in jpeg:

                filename = jpeg

                origin = os.path.join(current_directory +'/' + filename)

                destination = os.path.join(new_dir +'/' +filename)

                shutil.copy(origin, destination)

            else:

                pass

    

#check to see how many files are in each directory

train_dir = 'train/normal'

print('Total images in train/normal folder is {}'.format(len(os.listdir(train_dir))))



train_dir = 'train/bacteria'

print('Total images in train/bacteria folder is {}'.format(len(os.listdir(train_dir))))



train_dir = 'train/virus'

print('Total images in train/virus folder is {}'.format(len(os.listdir(train_dir))))



print('\n')



train_dir = 'val/normal'

print('Total images in train/normal folder is {}'.format(len(os.listdir(train_dir))))



train_dir = 'val/bacteria'

print('Total images in train/bacteria folder is {}'.format(len(os.listdir(train_dir))))



train_dir = 'val/virus'

print('Total images in val/virus folder is {}'.format(len(os.listdir(train_dir))))



folders = ['normal', 'bacteria','virus']

train = 'train/'

val = 'val/'



#Move about 10% of the files from the train folder to the validation folder

for folder in folders:    

    current_directory = train + folder

    dst_directory = val+folder

    

    current_dirList = os.listdir(current_directory)

    

    for img_count, img in enumerate(current_dirList):

        if img_count % 10 ==0:

            origin = os.path.join(current_directory + '/' + img)

            destination = os.path.join(dst_directory+'/'+img)

            shutil.move(origin, destination)
#Double check to make sure



train_dir = 'train/normal'

print('Total images in train/normal folder is {}'.format(len(os.listdir(train_dir))))



train_dir = 'train/bacteria'

print('Total images in train/bacteria folder is {}'.format(len(os.listdir(train_dir))))



train_dir = 'train/virus'

print('Total images in train/virus folder is {}'.format(len(os.listdir(train_dir))))



print('\n')



train_dir = 'val/normal'

print('Total images in train/normal folder is {}'.format(len(os.listdir(train_dir))))



train_dir = 'val/bacteria'

print('Total images in train/bacteria folder is {}'.format(len(os.listdir(train_dir))))



train_dir = 'val/virus'

print('Total images in val/virus folder is {}'.format(len(os.listdir(train_dir))))

#directory names

train_dir = 'train/'

test_dir = 'test/'

val_dir = 'val/'



# All images will be rescaled by 1./255

test_datagen = ImageDataGenerator(rescale=1./255)

val_datagen = ImageDataGenerator(rescale = 1./255)

train_datagen = ImageDataGenerator(rescale=1./255)



#set up generators

train_generator = train_datagen.flow_from_directory(

        train_dir,

        target_size=(150, 150),

        batch_size=20,

        class_mode='categorical')



validation_generator = val_datagen.flow_from_directory(

        val_dir,

        target_size=(150, 150),

        batch_size=20,

        class_mode='categorical')



test_generator = test_datagen.flow_from_directory(

        test_dir,

        target_size=(150, 150),

        batch_size=20,

        class_mode='categorical',

        shuffle=False)
#saves the best model weights based on the loss value; checkpoint path will be renamed for each different model if used

checkpoints_path = 'weights_base.best.hdf5'

checkpoint = ModelCheckpoint(checkpoints_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')



#stops the model if model does not improve after 10 epochs

early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=10)



callbacks = [checkpoint, early_stopping]
#create the model with Sequential

cnn_model = models.Sequential()

cnn_model.add(layers.Conv2D(32, (3, 3), activation='relu',

                        input_shape=(150 ,150,  3)))

cnn_model.add(layers.MaxPooling2D((2, 2)))



cnn_model.add(layers.Conv2D(64, (4, 4), activation='relu'))

cnn_model.add(layers.MaxPooling2D((2, 2)))



cnn_model.add(layers.Conv2D(128, (3, 3), activation='relu'))

cnn_model.add(layers.MaxPooling2D((2, 2)))



cnn_model.add(layers.Flatten())

cnn_model.add(layers.Dense(256, activation='relu'))



cnn_model.add(layers.Dense(3, activation='softmax'))
#compiles the model

cnn_model.compile(loss='categorical_crossentropy',

              optimizer="adam",

              metrics=['acc'])



cnn_model.summary()
#fits the model

cnn_history = cnn_model.fit_generator(

      train_generator,

      steps_per_epoch=10,

      epochs=20,

      validation_data=validation_generator,

      validation_steps=20,

    callbacks = [early_stopping])
#Creates two graphs of training/validation accuracy and loss over each epoch

def plot_acc_loss(history):

    train_acc = history.history['acc']

    val_acc = history.history['val_acc']

    train_loss = history.history['loss']

    val_loss = history.history['val_loss']

    epch = range(1, len(train_acc) + 1)

    plt.plot(epch, train_acc, 'g.', label='Training Accuracy')

    plt.plot(epch, val_acc, 'g', label='Validation acc')

    plt.title('Accuracy')

    plt.legend()

    plt.figure()

    plt.plot(epch, train_loss, 'r.', label='Training loss')

    plt.plot(epch, val_loss, 'r', label='Validation loss')

    plt.title('Loss')

    plt.legend()

    plt.show()
plot_acc_loss(cnn_history)
#prints the model accuracy compared to the test data

test_loss, test_acc = cnn_model.evaluate_generator(test_generator, steps=50)

print('test acc:', test_acc)

#load the pretrained model as the base

cnn_base = VGG19(weights='imagenet',

                  include_top=False,

                  input_shape=(150, 150, 3))
cnn_base.summary()
#create the model using the VGG19 pretrained as a base layer

pretrain_model = models.Sequential()

pretrain_model.add(cnn_base)

pretrain_model.add(layers.Flatten())

pretrain_model.add(layers.Dense(64, activation='relu'))

pretrain_model.add(layers.Dense(128, activation='relu'))

pretrain_model.add(layers.Dense(256, activation='relu'))

pretrain_model.add(layers.Dense(128, activation='relu'))

pretrain_model.add(layers.Dense(3, activation='softmax'))



pretrain_model.summary()
#Compilation

pretrain_model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['acc'])



#Fitting the Model

pretrain_history = model.fit_generator(

              train_generator,

              steps_per_epoch= 20,

              epochs = 20,

              validation_data = validation_generator,

              validation_steps = 10)
#plot graphs to check loss/accuracy of train/validation

plot_acc_loss(pretrain_history)
#prints the model accuracy compared to the test data

test_loss, test_acc = pretrain_model.evaluate_generator(test_generator, steps=50)

print('test acc:', test_acc)
cnn_base.summary()
#This function uses the pretrained network to extract the necessary features



datagen = ImageDataGenerator(rescale=1./255) 

batch_size = 20

classes = 3



def extract_features(directory, sample_amount):

    features = np.zeros(shape=(sample_amount, 4, 4, 512)) #creates 4-D numpy 0's array with sample amt in first layer

    labels = np.zeros(shape=(sample_amount, classes)) # creates 1-D numpy 0's array of labels with sample amount

    generator = datagen.flow_from_directory(

        directory, target_size=(150, 150), 

        batch_size = batch_size, 

        class_mode='categorical') #takes data from directory given

    i=0 #iteration

    for inputs_batch, labels_batch in generator:  #for every batch in the generator

        features_batch = cnn_base.predict(inputs_batch) #use base CNN model to extract features based on inputs for that batch

        features[i * batch_size : (i + 1) * batch_size] = features_batch #turns 0's in features to feature predicted by generator

        labels[i * batch_size : (i + 1) * batch_size] = labels_batch #turns 0's in labels to labels predicted by generator

        i = i + 1  #increase interation

        if i * batch_size >= sample_amount: #if number of features >= sample amount, stops the program

            break

    return features, labels



#use the function above on each directory and reshape the array

train_features, train_labels = extract_features(train_dir, 2500) 

validation_features, validation_labels = extract_features(val_dir, 500) 

test_features, test_labels = extract_features(test_dir, 600)



train_features = np.reshape(train_features, (2500, 4 * 4 * 512)) #reshape to 2-D array

validation_features = np.reshape(validation_features, (500, 4 * 4 * 512))

test_features = np.reshape(test_features, (600, 4 * 4 * 512))


fe_model = models.Sequential() #builds sequential model

fe_model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))

fe_model.add(layers.Dense(3, activation='softmax')) #converge to output layer



fe_model.compile(optimizer='adam', #compiles model

              loss='categorical_crossentropy',

              metrics=['acc'])



fe_history = fe_model.fit(train_features, train_labels, #fits model to train/valid feature/labels

                    epochs=20,

                    batch_size=20,

                    validation_data=(validation_features, validation_labels),

                    callbacks = [early_stopping])
plot_acc_loss(fe_history)
test_loss, test_acc = fe_model.evaluate(test_features, test_labels)#eval model using test feats/labels

print('test acc:', test_acc)

#Rerun the feature extraction model with few epochs to prevent overfitting

fe_history = fe_model.fit(train_features, train_labels, #fits model to train/valid feature/labels

                    epochs=3,

                    batch_size=20,

                    validation_data=(validation_features, validation_labels),

                    callbacks = [early_stopping])



test_loss, test_acc = fe_model.evaluate(test_features, test_labels)#eval model using test feats/labels

print('test acc:', test_acc)
cnn_base = VGG19(weights='imagenet',

                  include_top=False,

                  input_shape=(150, 150, 3))
#build the model

freeze_model = models.Sequential()

freeze_model.add(cnn_base)

freeze_model.add(layers.Flatten())

freeze_model.add(layers.Dense(132, activation='relu'))

freeze_model.add(layers.Dense(3, activation='softmax'))



#freeze the base so weights can't be changed

cnn_base.trainable = False



#compile

freeze_model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['acc'])

#name of the file for the best weights

checkpoints_path = 'freeze_weights_base.best.hdf5'

checkpoint = ModelCheckpoint(checkpoints_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
freeze_history = freeze_model.fit_generator(

              train_generator,

              steps_per_epoch =20,

              epochs = 20,

              validation_data = validation_generator,

              validation_steps = 10,

              callbacks = [checkpoint])
plot_acc_loss(freeze_history)
test_loss, test_acc = freeze_model.evaluate_generator(test_generator, steps=50)

print('test acc:', test_acc)
freeze_model = models.Sequential()

freeze_model.add(cnn_base)

freeze_model.add(layers.Flatten())

freeze_model.add(layers.Dense(132, activation='relu'))

freeze_model.add(layers.Dense(3, activation='softmax'))





freeze_model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['acc'])



#load the best weights into the model

freeze_model.load_weights('freeze_weights_base.best.hdf5')



test_loss, test_acc = freeze_model.evaluate_generator(test_generator, steps=50)

print('test acc:', test_acc)
freeze_model.summary()
#check if base is frozen

cnn_base.trainable
#function that takes a cnn base along with the name of a layer. Then, unfreezes every layer from the specified layer onwards



def make_trainable(cnn_base, layer_name):

    cnn_base.trainable = True

    set_trainable = False

    for layer in cnn_base.layers:  #sets every layer from the specified layer to the last layer as trainable. 

        if layer.name == layer_name:

            set_trainable = True

        if set_trainable:

            layer.trainable = True

        else:

            layer.trainable = False



#makes the last 5 layers trainable

make_trainable(cnn_base, 'block5_conv1')
#check if the base has some unfrozen layers

cnn_base.summary()
freeze_model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['acc'])



ft_history = freeze_model.fit_generator(

              train_generator, 

              steps_per_epoch= 20,

              epochs = 20,

              validation_data = validation_generator,

              validation_steps = 10)
plot_acc_loss(ft_history)
#prints test accuracy

test_loss, test_acc = freeze_model.evaluate_generator(test_generator, steps=50)

print('test acc:', test_acc)
better_model = models.Sequential()



better_model.add(layers.Conv2D(32, (3, 3), activation='relu',

                        input_shape=(150 ,150,  3)))

better_model.add(layers.MaxPooling2D((2, 2)))



better_model.add(layers.Conv2D(64, (4, 4), activation='relu'))

better_model.add(layers.Conv2D(64, (4, 4), activation='relu'))

better_model.add(layers.MaxPooling2D((2, 2)))



better_model.add(layers.Conv2D(128, (3, 3), activation='relu'))

better_model.add(layers.Conv2D(128, (3, 3), activation='relu'))

better_model.add(layers.MaxPooling2D((2, 2)))





better_model.add(layers.Flatten())

better_model.add(layers.Dense(256, activation='relu'))

better_model.add(layers.Dense(512, activation='relu'))

better_model.add(layers.Dense(64, activation='relu'))

better_model.add(layers.Dense(128, activation='relu'))

better_model.add(layers.Dense(3, activation='softmax'))



better_model.summary()
checkpoints_path = 'better_cnn_weights.best.hdf5'

checkpoint = ModelCheckpoint(checkpoints_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
better_model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['acc'])



better_history = better_model.fit_generator(

              train_generator, 

              steps_per_epoch= 20,

              epochs = 20,

              validation_data = validation_generator,

              validation_steps = 10,

              callbacks = [checkpoint])
plot_acc_loss(better_history)
test_loss, test_acc = better_model.evaluate_generator(test_generator, steps=50)

print('test acc:', test_acc)
#build the model again

better_model = models.Sequential()



better_model.add(layers.Conv2D(32, (3, 3), activation='relu',

                        input_shape=(150 ,150,  3)))

better_model.add(layers.MaxPooling2D((2, 2)))



better_model.add(layers.Conv2D(64, (4, 4), activation='relu'))

better_model.add(layers.Conv2D(64, (4, 4), activation='relu'))

better_model.add(layers.MaxPooling2D((2, 2)))



better_model.add(layers.Conv2D(128, (3, 3), activation='relu'))

better_model.add(layers.Conv2D(128, (3, 3), activation='relu'))

better_model.add(layers.MaxPooling2D((2, 2)))





better_model.add(layers.Flatten())

better_model.add(layers.Dense(256, activation='relu'))

better_model.add(layers.Dense(512, activation='relu'))

better_model.add(layers.Dense(64, activation='relu'))

better_model.add(layers.Dense(128, activation='relu'))

better_model.add(layers.Dense(3, activation='softmax'))



#compile it again

better_model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['acc'])



#load weights

better_model.load_weights('weights_base.best.hdf5')



#evaluate the model

test_loss, test_acc = better_model.evaluate_generator(test_generator, steps=50)

print('test acc:', test_acc)