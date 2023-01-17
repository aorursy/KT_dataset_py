### MNIST Digit Classification via KERAS using Fully-Connected Neural Network

##### A simple 2 layer fully-connected feed forward neural network that achieves ~99.998% on training data set and ~97.7 on the test data set.

# Import Numpy, TensorFlow, Keras and vectorized MNIST data
import numpy as np
from numpy import array
#import tensorflow as tf
import keras
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical

# Load the training set:
train_set = pd.read_csv('../input/train.csv')
# extract the labels:
train_label = train_set.label
# training features, normalized:
train_feat = np.array(train_set.iloc[:, 1:])/255
# reshape the column vector to row array
train_label = np.array(train_label).reshape(-1, 1)
# one hot encode labels:
encoded_label = to_categorical(train_label)

# Define the Sequential Feed-farward Model
model = Sequential() 
model.add(Dense(units=300, input_dim=784, activation='relu'))
model.add(Dense(units=200, activation='relu'))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=10))
model.add(Activation('softmax'))

# Initialize and compile:
keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=None)
model.compile(optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True), loss='categorical_crossentropy')
#model.compile(optimizer=Adam(lr=0.001, decay=1e-6), loss='categorical_crossentropy')

# Training
history = model.fit(train_feat, encoded_label, validation_split=0.01,  batch_size=64, epochs=15, verbose=0)
# Find the indices of the most confident prediction for each item. That tells us the predicted digit for that sample.
predictions = model.predict(np.array(train_feat)).argmax(axis=1)
actual = train_label[:,0]
# Calculate the accuracy, which is the percentage of times the predicated labels matched the actual labels
train_accuracy = np.mean(predictions == actual)

# Print out the result
print("Train accuracy: ", train_accuracy)

# Plot some figures:
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

plt.semilogy(history.history['loss'])
plt.semilogy(history.history['val_loss'])
plt.title('Model Complexity Graph:  Training vs. Validation Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='upper right')

# Load the test set:
test_set = pd.read_csv('../input/test.csv')
# test features:
test_feat = np.array(test_set)/255
predictions = model.predict(np.array(test_feat)).argmax(axis=1)

print('Done!')
