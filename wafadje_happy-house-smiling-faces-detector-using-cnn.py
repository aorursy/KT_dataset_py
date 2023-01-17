# import libraries 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O

import matplotlib.pyplot as plt # Import matplotlib for data visualisation

import seaborn as sns

import h5py #It lets you store huge amounts of numerical data, and easily manipulate that data from NumPy

import random 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
filename = '/kaggle/input/train_happy.h5'

f = h5py.File(filename, 'r')



for key in f.keys():

    print(key) #Names of the groups in HDF5 file.
happy_training = h5py.File('/kaggle/input/train_happy.h5', "r")

happy_testing  = h5py.File('/kaggle/input/test_happy.h5', "r")
X_train = np.array(happy_training["train_set_x"][:]) 

y_train = np.array(happy_training["train_set_y"][:]) 



X_test = np.array(happy_testing["test_set_x"][:])

y_test = np.array(happy_testing["test_set_y"][:]) 



#print(X_train, y_train, X_test, y_test)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
i = random.randint(1,600) # select any random index from 1 to 600

plt.imshow( X_train[i] )

print(y_train[i])
W_grid = 3

L_grid = 3



fig, axes = plt.subplots(L_grid, W_grid, figsize = (9,9))



axes = axes.ravel() # flaten the 15 x 15 matrix into 225 array



n_training = len(X_train) # get the length of the training dataset



for i in np.arange(0, W_grid * L_grid): # create evenly spaces variables 



    index = np.random.randint(0, n_training)

    axes[i].imshow( X_train[index])

    axes[i].set_title(y_train[index], fontsize = 25)

    axes[i].axis('off')



plt.subplots_adjust(hspace=0.4)

# Let's normalize dataset

X_train = X_train/255

X_test = X_test/255
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

from keras.optimizers import Adam

from keras.callbacks import TensorBoard
cnn_model = Sequential() #Specifying the input shape



cnn_model.add(Conv2D(64, 6, 6, input_shape = (64,64,3), activation='relu'))

cnn_model.add(MaxPooling2D(pool_size = (2, 2)))



cnn_model.add(Dropout(0.2))



cnn_model.add(Conv2D(64, 5, 5, activation='relu'))

cnn_model.add(MaxPooling2D(pool_size = (2, 2)))



cnn_model.add(Flatten())

cnn_model.add(Dense(output_dim = 128, activation = 'relu'))

cnn_model.add(Dense(output_dim = 1, activation = 'sigmoid'))
cnn_model.compile(loss ='binary_crossentropy', optimizer=Adam(lr=0.001),metrics =['accuracy'])
epochs = 9



history = cnn_model.fit(X_train,

                        y_train,

                        batch_size = 60,

                        nb_epoch = epochs,

                        verbose = 1)
evaluation = cnn_model.evaluate(X_test, y_test)

print('Test Accuracy : {:.3f}'.format(evaluation[1]))
# get the predictions for the test data

predicted_classes = cnn_model.predict_classes(X_test)

print("predicted_classes shape : ", predicted_classes.shape)

print("y_test shape : ", y_test.shape)
L = 5

W = 5

fig, axes = plt.subplots(L, W, figsize = (12,12))

axes = axes.ravel() # 



for i in np.arange(0, L * W):  

    axes[i].imshow(X_test[i])

    axes[i].set_title("Prediction Class = {}\n True Class = {}".format(predicted_classes[i], y_test[i]))

    axes[i].axis('off')



plt.subplots_adjust(wspace=0.5)
from sklearn.metrics import classification_report



print(classification_report(y_test.T, predicted_classes))