from keras.models import Sequential 

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import Flatten

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.utils import np_utils

from keras.datasets import reuters

from keras import models

from keras import layers

from keras.datasets import mnist

from keras.models import model_from_json

from keras import backend as K

K.set_image_dim_ordering('th')

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import cv2

import os



# load data



# Train ===========================

train_data = pd.read_csv("train.csv")

x_train = train_data.drop(labels = ["label"],axis = 1) 

y_train = train_data["label"]



# Test ===========================

test_data = pd.read_csv("test.csv")

#check the missing data in the train set 

sum_cols_x = x_train.isnull().sum(axis=1)

sum_all_x = sum_cols_x.sum()

print(sum_all_x)



sum_all_y = y_train.isnull().sum()

print(sum_all_y)
#check the missing data in the test set 

sum_cols_test = test_data.isnull().sum(axis=1)

sum_all_test = sum_cols_test.sum()

print(sum_all_test)
print(x_train.shape)

print(y_train.shape)

print(test_data.shape)
x_train = x_train.to_numpy()

y_train = y_train.to_numpy()

x_test = test_data.to_numpy()
# Reshape to be suitable for the visualization of some images 

x_train = x_train.reshape(x_train.shape[0], 28, 28).astype('float32')

x_test = x_test.reshape(x_test.shape[0], 28, 28).astype('float32')
# plot 4 images as gray scale

plt.subplot(221)

plt.imshow(x_train[0], cmap=plt.get_cmap('gray'))

plt.subplot(222)

plt.imshow(x_train[1], cmap=plt.get_cmap('gray'))

plt.subplot(223)

plt.imshow(x_test[227], cmap=plt.get_cmap('gray'))

plt.subplot(224)

plt.imshow(x_test[30], cmap=plt.get_cmap('gray'))

# show the plot

plt.show()
# Reshape to be suitable for the cnn

# reshape to be [samples][channel][width][height]

x_train = x_train.reshape(x_train.shape[0], 28, 28,1).astype('float32')

x_test = x_test.reshape(x_test.shape[0], 28, 28,1).astype('float32')





y_train = y_train.reshape(y_train.shape[0], 1).astype('float32')





# normalize inputs from 0-255 to 0-1

x_train = x_train / 255

x_test = x_test / 255



# one hot encode outputs

y_train = np_utils.to_categorical(y_train)

num_classes = y_train.shape[1]

print(num_classes)
print(x_train.shape)

print(x_test.shape)

print(y_train.shape)
# define the larger model

def cnn_model():

    # create model

    model = Sequential()

    model.add(Conv2D(30, (5, 5), input_shape=(28,28,1), activation='relu',data_format="channels_last"))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(15, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))

    model.add(Dense(50, activation='relu'))

    model.add(Dense(num_classes, activation='softmax'))

    # Compile model

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
# build the model

model = cnn_model()

# Fit the model

history = model.fit(x_train, y_train,validation_split = 0.1, epochs=20, batch_size=200)
plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
index = 227 # of the sample to be tested

y = model.predict_classes(x_test[index:index+1])

print(y)
model.save('new_digits_model.h5')
 

# load json and create model

json_file = open('model.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)

# load weights into new model

loaded_model.load_weights("model.h5")

print("Loaded model from disk")
output = model.predict_classes(x_test)
print(output)
image_ids = range(1, len(x_test)+1)

submission_df = pd.DataFrame({'ImageId': image_ids, 'Label': output})

#Saving the final dataframe as csv, which can be submitted now.

submission_df.to_csv(path_or_buf='digits_submission.csv', index=False)