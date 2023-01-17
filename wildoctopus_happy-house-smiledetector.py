#NOTE: This is just a very basic model but will update it slightely when I have time, to get more accurate prediction. If you like it please upvote and if you have any question please feel free to ask in comment box.

#Happy House Dataset
#Detect whether a person is smiling or not!
#Using CNN
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy


import keras 
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Activation, Dropout
from keras.optimizers import Adam, RMSprop
from keras.layers.normalization import BatchNormalization

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score

import seaborn as sns
#function to load data
def load_dataset():
    train_data = h5py.File('../input/train_happy.h5', "r")
    x_train = np.array(train_data["train_set_x"][:]) 
    y_train = np.array(train_data["train_set_y"][:]) 

    test_data = h5py.File('../input/test_happy.h5', "r")
    x_test = np.array(test_data["test_set_x"][:])
    y_test = np.array(test_data["test_set_y"][:]) 
    
    y_train = y_train.reshape((1, y_train.shape[0]))
    y_test = y_test.reshape((1, y_test.shape[0]))
    
    return x_train, y_train, x_test, y_test
# Load the data
X_train, Y_train, X_test, Y_test = load_dataset()
# Example of an image
index = 1
plt.imshow(X_train[index])
print ("y = " + str(np.squeeze(Y_train[:, index])))
#check shape of our data
print (X_train.shape)
print (X_test.shape)
print (Y_train.shape)
print (Y_test.shape)
#Rescale data
X_train = X_train/255.
X_test = X_test/255.
Y_train = Y_train.T
Y_test = Y_test.T
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
#Updated CNN model  --- better Accuracy n precesion.

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 
                 input_shape = (64,64,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))


#Output Layer
model.add(Dense(units = 1,kernel_initializer="uniform", activation = 'sigmoid'))

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# Compiling Neural Network
model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(X_train, Y_train, batch_size=20, epochs=35)
# Predict the test set results
Y_pred = model.predict_classes(X_test)
print ("test accuracy: %s" %accuracy_score(Y_test, Y_pred))
print ("precision: %s"  %precision_score(Y_test, Y_pred))
print ("recall: %s" %recall_score(Y_test, Y_pred))
print ("f1 score: %s"  %f1_score(Y_test, Y_pred))


cm = confusion_matrix(Y_test,Y_pred)
sns.heatmap(cm,annot=True)
# **Note : TuneUp the parameters to get more accuracy. Add Data Augmentation. 