# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # Import matplotlib for data visualisation

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
train.shape
test.shape
train
train.info()
train = np.array(train, dtype='float32')

test = np.array(test, dtype='float32')
print(train)
import random

# Let's view some images!

i = random.randint(1,42000) # select any random index from 1 to 60,000

label = train[i,0]

print(label)

plt.imshow( train[i,1:].reshape((28,28)), cmap = 'gray' ) # reshape and plot the image


# Let's view more images in a grid format

# Define the dimensions of the plot grid 

W_grid = 15

L_grid = 15



# fig, axes = plt.subplots(L_grid, W_grid)

# subplot return the figure object and axes object

# we can use the axes object to plot specific figures at various locations



fig, axes = plt.subplots(L_grid, W_grid, figsize = (17,17))



axes = axes.ravel() # flaten the 15 x 15 matrix into 225 array



n_training = len(train) # get the length of the training dataset



# Select a random number from 0 to n_training

for i in np.arange(0, W_grid * L_grid): # create evenly spaces variables 



    # Select a random number

    index = np.random.randint(0, n_training)

    # read and display an image with the selected index    

    axes[i].imshow( train[index,1:].reshape((28,28)) )

    axes[i].set_title(train[index,0], fontsize = 8)

    axes[i].axis('off')



plt.subplots_adjust(hspace=0.4)
X_train = train[:,1:]/255

X_test = test[:,:]/255



y_train = train[:,0]
import keras

y_train = keras.utils.to_categorical(y_train, 10)
X_train.shape
y_train.shape
from sklearn.model_selection import train_test_split



X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size = 0.1, random_state = 12345)
X_train.shape
X_validate.shape
X_test.shape
y_train.shape
y_validate.shape
# * unpack the tuple

X_train = X_train.reshape(X_train.shape[0], *(28, 28, 1))

X_test = X_test.reshape(X_test.shape[0], *(28, 28, 1))

X_validate = X_validate.reshape(X_validate.shape[0], *(28, 28, 1))
X_train.shape
X_validate.shape
X_test.shape
import tensorflow.keras as keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense

from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.preprocessing.image import ImageDataGenerator


model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))





model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))





model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))
model.summary()
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(loss ='categorical_crossentropy', optimizer=optimizer ,metrics =['accuracy'])
# Set a learning rate annealer

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
# With data augmentation to prevent overfitting (accuracy 0.99286)

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
epochs = 30

history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=86),

                           epochs = epochs,

                           verbose = 2,

                           validation_data = (X_validate, y_validate),

                           callbacks=[learning_rate_reduction])
# get the predictions for the test data

results = model.predict(X_test)

# select the indix with the maximum probability

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")
results
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn_mnist_datagen.csv",index=False)
# Plot the loss and accuracy curves for training and validation 

fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)