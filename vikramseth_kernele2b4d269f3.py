# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import csv

import math as m

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from keras import layers

from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D

from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D

from keras.models import Model

from keras.callbacks import ReduceLROnPlateau

from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

import os

import time

import matplotlib.pyplot as plt

%matplotlib inline

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots

plt.rcParams['image.interpolation'] = 'nearest'

plt.rcParams['image.cmap'] = 'gray'

print(os.listdir("../input"))

np.random.seed(2)

# Any results you write to the current directory are saved as output.
def convert_to_one_hot(Y, C):

    Y = np.eye(C)[Y.reshape(-1)]

    return Y
def read_csv(filename):

    X,Y=[],[]

    test=pd.read_csv(filename)

    if filename.find("train.csv") >0:

                Y=test.iloc[:,0].values

                Y=convert_to_one_hot(Y,10)

                X=test.iloc[:,1:785].values

    else:

                X=test.iloc[:,0:784].values

    

    return X, Y
def write_csv(filename,predictions):

# Writing a CSV file submission. ImageName,Prediction 



    my_submission = pd.DataFrame({'ImageId': range(1,predictions.shape[0]+1), 'Label': predictions})

    # you could use any filename. We choose submission here

    my_submission.to_csv('submission.csv', index=False)

            
X_train,Y_train=read_csv("../input/train.csv")

X_test,_=read_csv("../input/test.csv")

m,pixels=X_train.shape

classes=10 

height,width,channels=28,28,1

X_train, X_test=X_train/255, X_test/255

#plt.imshow(np.reshape(X_test[0],(28,28)))   #plotting the example data

#Resize into (height width channels)

X_train=X_train.reshape(-1,height,width,channels)

X_test=X_test.reshape(-1,height,width,channels)

print(Y_train.shape,X_train.shape,X_test.shape)

# Split the train and the validation set for the fitting

random_seed = 2

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)

print(Y_train.shape,X_train.shape,X_test.shape,X_val.shape,Y_val.shape)
def DigitalRecognizerModel(input_shape):

    """

       X_train size[None,height,width,channels]

       Y_train size[None,classes]

       X_test size[None,height,width,channels]

    """

    # Define the inputpadding = 'same', placeholder as a tensor with shape input_shape. Think of this as your input image!

    X_input = Input(input_shape)

    X=Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1))(X_input)

    X=Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu')(X)

    X=MaxPooling2D(pool_size=(2,2))(X)

    X=Dropout(0.25)(X)



    X=Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu')(X)

    X=Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu')(X)

    X=MaxPooling2D(pool_size=(2,2), strides=(2,2))(X)

    X=Dropout(0.25)(X)

    X=Flatten()(X)

    X=Dense(256, activation = "relu")(X)

    X=Dropout(0.5)(X)

    X=Dense(10, activation = "softmax")(X)

    # Create model

    model = Model(inputs = X_input, outputs = X, name='DigitalRecognizer')    

    return model
datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images





datagen.fit(X_train)
# Set a learning rate annealer

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
digitalRecognizerModel = DigitalRecognizerModel(X_train[0].shape)

digitalRecognizerModel.compile(optimizer = "Adam", loss = "binary_crossentropy", metrics = ["accuracy"])

#digitalRecognizerModel.fit(x=X_train, y=Y_train, epochs=10, batch_size=62)

history = digitalRecognizerModel.fit_generator(datagen.flow(X_train,Y_train, batch_size=62),

                              epochs = 30, validation_data = (X_val,Y_val),

                              verbose = 2, steps_per_epoch=610,callbacks=[learning_rate_reduction])  #m.ceil(X_train.shape[0] // 62)
print("Time Start:" ,time.time())



val_predictions=digitalRecognizerModel.predict(X_val)

#test Accuracy

correct_val_predictions=np.mean(np.equal(np.argmax(val_predictions,axis=1), np.argmax(Y_val,axis=1)))

print("Validation Accuracy",correct_val_predictions)

#test Predictions

test_predictions=digitalRecognizerModel.predict(X_test)

correct_test_predictions=np.argmax(test_predictions,axis=1)

write_csv('submission.csv',correct_test_predictions)

print("Time End:" ,int(round(time.time())))