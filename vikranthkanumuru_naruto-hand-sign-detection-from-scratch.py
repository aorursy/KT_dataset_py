# Dont run this













#Image Augumentation















from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing.image import img_to_array

from keras.preprocessing.image import load_img

import numpy as np

import argparse

# construct the argument parser and parse the arguments





# load the input image, convert it to a NumPy array, and then

# reshape it to have an extra dimension

for i in os.listdir(mainPath):

    if not j.startswith('.'):

        count = 0 

        for j in os.listdir(mainPath + str(i) + '/'):

            print("[INFO] loading example image...")

            image = load_img(j)

            image = img_to_array(image)

            image = np.expand_dims(image, axis=0)

            # construct the image generator for data augmentation then

            # initialize the total number of images generated thus far

            aug = ImageDataGenerator(

                rotation_range=30,

                zoom_range=0.15,

                width_shift_range=0.2,

                height_shift_range=0.2,

                shear_range=0.15,

                horizontal_flip=True,

                fill_mode="nearest")

            total = 0

            # construct the actual Python generator

            print("[INFO] generating images...")

            imageGen = aug.flow(image, batch_size=1, save_to_dir=args["output"],

                save_prefix="image", save_format="jpg")

            # loop over examples from our image data augmentation generator

            for image in imageGen:

                # increment our counter

                total += 1

                # if we have reached the specified number of examples, break

                # from the loop

                if total == args["total"]:

                    break
%env JOBLIB_TEMP_FOLDER=/tmp
# preprocess.py

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os

import cv2

import random

import numpy as np

import keras

from random import shuffle

from keras.utils import np_utils

from shutil import unpack_archive

print("Imported Modules...")

mainPath="/kaggle/input/naruto-hand-sign-dataset/Pure Naruto Hand Sign Dataset/"

# data folder path

data_folder_path = mainPath 

files = os.listdir(data_folder_path)



# dictionary to maintain numerical labels

countForDicCount=0

class_dict = {}

# dictionary to maintain counts

class_count = {}



for i in os.listdir(mainPath):

    class_dict[i]=countForDicCount

    class_count[i]=0

    countForDicCount+=1

    



            

# training lists

X = []

Y = []

# validation lists

X_val = []

Y_val = []

# testing lists

X_test = []

Y_test = []





for i in os.listdir(mainPath):

    if not i.startswith('.'):

        label=i

        for j in os.listdir(mainPath + str(i) + '/'):

            if label in class_dict:

                path = mainPath + str(i) + '/'+str(j)

                image = cv2.imread(path)

                resized_image = cv2.resize(image,(224,224))

                if class_count[label]<10:

                  class_count[label]+=1

                  X.append(resized_image)

                  Y.append(class_dict[label])

                elif class_count[label]>=10 and class_count[label]<13:

                  class_count[label]+=1

                  X_val.append(resized_image)

                  Y_val.append(class_dict[label])

                else:

                  X_test.append(resized_image)

                  Y_test.append(class_dict[label])

            



# one-hot encodings of the classes

Y = np_utils.to_categorical(Y)

Y_val = np_utils.to_categorical(Y_val)

Y_test = np_utils.to_categorical(Y_test)  

  

if not os.path.exists('train_test_split'):

    os.makedirs('train_test_split')

    

npy_data_path="/kaggle/working/train_test_split"

np.save(npy_data_path+'/train_set.npy',X)

np.save(npy_data_path+'/train_classes.npy',Y)

np.save(npy_data_path+'/validation_set.npy',X_val)

np.save(npy_data_path+'/validation_classes.npy',Y_val)

np.save(npy_data_path+'/test_set.npy',X_test)

np.save(npy_data_path+'/test_classes.npy',Y_test)

print("Data pre-processing Success!")





X= np.asarray(X, dtype=np.float32)



print("This is the X.shape {}".format(X.shape))

print("This is the Y.shape {}".format(Y.shape))



# training.py

if not os.path.exists('Weights_Full'):

    os.makedirs('Weights_Full')

    

    

if not os.path.exists('Checkpoint'):

    os.makedirs('Checkpoint')









from keras.optimizers import SGD

from keras.models import Sequential

from keras.preprocessing import image

from keras.layers.normalization import BatchNormalization

from keras.layers import Dense, Activation, Dropout, Flatten,Conv2D, MaxPooling2D

print("Imported Network Essentials")

# loading .npy dataset

X_train=np.load(npy_data_path+"/train_set.npy")

Y_train=np.load(npy_data_path+"/train_classes.npy")

X_valid=np.load(npy_data_path+"/validation_set.npy")

Y_valid=np.load(npy_data_path+"/validation_classes.npy")

X_test=np.load(npy_data_path+"/test_set.npy")

Y_test=np.load(npy_data_path+"/test_classes.npy")

X_test.shape



model = Sequential()

# 1st Convolutional Layer

model.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11),strides=(4,4), padding='valid'))

model.add(Activation('relu'))

# Pooling 

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# Batch Normalisation before passing it to the next layer

model.add(BatchNormalization())

# 2nd Convolutional Layer

model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))

model.add(Activation('relu'))

# Pooling

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# Batch Normalisation

model.add(BatchNormalization())

# 3rd Convolutional Layer

model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))

model.add(Activation('relu'))

# Batch Normalisation

model.add(BatchNormalization())

# 4th Convolutional Layer

model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))

model.add(Activation('relu'))

# Batch Normalisation

model.add(BatchNormalization())

# 5th Convolutional Layer

model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))

model.add(Activation('relu'))

# Pooling

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# Batch Normalisation

model.add(BatchNormalization())

# Passing it to a dense layer

model.add(Flatten())

# 1st Dense Layer

model.add(Dense(4096, input_shape=(224*224*3,)))

model.add(Activation('relu'))

# Add Dropout to prevent overfitting

model.add(Dropout(0.4))

# Batch Normalisation

model.add(BatchNormalization())

# 2nd Dense Layer

model.add(Dense(4096))

model.add(Activation('relu'))

# Add Dropout

model.add(Dropout(0.6))

# Batch Normalisation

model.add(BatchNormalization())

# 3rd Dense Layer

model.add(Dense(1000))

model.add(Activation('relu'))

# Add Dropout

model.add(Dropout(0.5))

# Batch Normalisation

model.add(BatchNormalization())

# Output Layer

model.add(Dense(13))

model.add(Activation('softmax'))

model.summary()

# (4) Compile 

sgd = SGD(lr=0.001)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

##checkpoint = keras.callbacks.ModelCheckpoint("Checkpoint/weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

# (5) Train

##model.fit(X_train/255.0, Y_train, batch_size=32, epochs=50, verbose=1,validation_data=(X_valid/255.0,Y_valid/255.0), shuffle=True,callbacks=[checkpoint])

model.fit(X_train/255.0, Y_train, batch_size=32, epochs=50, verbose=1,validation_data=(X_valid/255.0,Y_valid/255.0), shuffle=True)



# serialize model to JSON

if not os.path.exists('Weights_Full'):

    os.makedirs('Weights_Full')

    

model_data_path="/kaggle/working/Weights_Full"

model_json = model.to_json()

with open(model_data_path+"/model.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

model.save_weights(model_data_path+"/model_weights.h5")

print("Saved model to disk")

# Compile 

sgd = SGD(lr=0.001)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

checkpoint = keras.callbacks.ModelCheckpoint("Checkpoint/weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=0, 

save_best_only=False, save_weights_only=False, mode='auto', period=1)
# serialize model to JSON

if not os.path.exists('Weights_Full'):

    os.makedirs('Weights_Full')

    

model_data_path="/kaggle/working/Weights_Full"

model_json = model.to_json()

with open(model_data_path+"/model.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

model.save_weights(model_data_path+"/model_weights.h5")

print("Saved model to disk")
# test.py

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) 

from keras.preprocessing import image

import numpy as np

from keras.models import model_from_json

from sklearn.metrics import accuracy_score 

# dimensions of our images

image_size = 224 

# load the model in json format

model_data_path="/kaggle/working/Weights_Full"

with open(model_data_path+'/model.json', 'r') as f:

    model = model_from_json(f.read())

    #model.summary()

model.load_weights(model_data_path+'/model_weights.h5')

model.load_weights('Weights/weights.250-0.00.hdf5') 

X_test=np.load(npy_data_path+"/test_set.npy")

Y_test=np.load(npy_data_path+"/test_classes.npy")

Y_predict = model.predict(X_test)

Y_predict = [np.argmax(r) for r in Y_predict]

Y_test = [np.argmax(r) for r in Y_test] 

print("##################")

acc_score = accuracy_score(Y_test, Y_predict)

print("Accuracy: " + str(acc_score))

print("##################")