#Disclaimer: I didn't make a lot of this and followed a tutorial to make this (like the CNN model).

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools

from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau
# Load the data

train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")
y = train["label"]

X = train.drop(labels = ["label"],axis = 1) 



sns.countplot(y)
# Turn everything to grayscale

X = X / 255.0

test = test / 255.0
# turn 784 list into 3D 28*28*1 matrix

X = X.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
#Encoder -> turns output of [0,0,1,0,0,0,0,0,0,0] to 2

y = to_categorical(y, num_classes = 10)
# Split the train and validation set

X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size = 0.1, random_state=2)
#CNN Model



CNN = Sequential()



CNN.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1)))

CNN.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))

CNN.add(MaxPool2D(pool_size=(2,2)))

CNN.add(Dropout(0.25))





CNN.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))

CNN.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))

CNN.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

CNN.add(Dropout(0.25))





CNN.add(Flatten())

CNN.add(Dense(256, activation = "relu"))

CNN.add(Dropout(0.5))

CNN.add(Dense(10, activation = "softmax"))
#Basically the optimizer is something that determines how well the model does. RMSprop basically creates the loss function

#to see how well it did

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# Compile the model

CNN.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
# Set a learning rate annealer - basically this determines how much/fast the model should change. patience is the number of epochs

# of negligible change. In this case, when it hits 4 epochs without change, the change rate will decreace to a factor of 0.6

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
epochs = 30

batch_size = 98 #8 batches
#Data Augmentation -> adds variation to increase sample size. This keeps from overfitting.



datagen = ImageDataGenerator(featurewise_center=False,samplewise_center=False,featurewise_std_normalization=False,

                             samplewise_std_normalization=False,zca_whitening=False,rotation_range=10,zoom_range = 0.1,width_shift_range=0.1,

                             height_shift_range=0.1,horizontal_flip=False,vertical_flip=False)

datagen.fit(X_train)
#Runs model

history = CNN.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),epochs = epochs, validation_data = (X_val,Y_val),

                            verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size, callbacks=[learning_rate_reduction])


#predicts results and turns it into pdf

results = CNN.predict(test)

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("submission.csv",index=False)