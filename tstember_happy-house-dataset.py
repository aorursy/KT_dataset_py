#Happy House Dataset
#Detect whether a person is smiling or not!
#Using CNN
import numpy as np
import h5py

import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy


import keras 
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Activation, Dropout
from keras.optimizers import Adam
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
#our CNN model

model = Sequential()

#Conv layer 1 
#input 64x64x3
model.add(Conv2D(64,(5,5), padding = 'Same', activation = 'relu', input_shape = (64,64,3)))
model.add(BatchNormalization())

#Pooling layer 1
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.1))

#Conv layer 2
model.add(Conv2D(64, (5,5), padding = 'Same', activation = 'relu'))
model.add(BatchNormalization())


#Pooling layer 2
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.1))

#Flatten
model.add(Flatten())

#Fully connected layer 1
model.add(Dense(128, activation = 'relu'))

#Output Layer
model.add(Dense(units = 1,kernel_initializer="uniform", activation = 'sigmoid'))


# Compiling Neural Network
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(X_train, Y_train, batch_size=20, epochs=35)
# Predict the test set results
Y_pred = model.predict_classes(X_test)
print ("test accuracy: %s" %accuracy_score(Y_test, Y_pred))
print ("precision: %s"  %precision_score(Y_test, Y_pred))
print ("recall: %s" %recall_score(Y_test, Y_pred))
print ("f1 score: %s"  %f1_score(Y_test, Y_pred))


cm = confusion_matrix(Y_test,Y_pred)
sns.heatmap(cm,annot=True)
