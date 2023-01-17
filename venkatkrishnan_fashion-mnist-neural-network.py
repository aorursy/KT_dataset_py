# Basic libs
import pandas as pd
import numpy as np

import os

import matplotlib.pyplot as plt

# SKlearn package libraries
from sklearn.model_selection import train_test_split

# Tensorflow libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential

print(f"Tensorflow version: {tf.__version__} ")
print(f"Tensorflow eager execution: {tf.executing_eagerly()}")
train_file = "/kaggle/input/fashionmnist/fashion-mnist_train.csv"
test_file = "/kaggle/input/fashionmnist/fashion-mnist_test.csv"
train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)
print(" Train data : {}, Test data : {}".format(train_data.shape, test_data.shape))

train_data.head()
train_df = np.array(train_data, dtype=np.float32)
test_df = np.array(test_data, dtype=np.float32)

# Normalizing the data
X_train = train_df[:, 1:] / 255.0
X_test = test_df[:, 1:] / 255.0

# Label data
y_train, y_test = np.array(train_df[:, 0], dtype=np.int32), np.array(test_df[:, 0], dtype=np.int32)

print(" Train data : {}, Test data : {}".format(X_train.shape, X_test.shape))
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
print(class_names[y_train[0]])
plt.figure(figsize=(12,12))

for i in range(36):
    plt.subplot(6,6,i+1)
    plt.imshow(X_train[i].reshape(28,28))
    index = y_train[i]
    plt.title(class_names[index])
    plt.xticks([])
    plt.yticks([])
plt.show()
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
# Reshape the pixels to 28x28 shape images
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_valid = X_valid.reshape(X_valid.shape[0], 28, 28, 1)

def get_ffnn():
    model = Sequential([
        keras.layers.Flatten(input_shape=[28,28,1]),
        keras.layers.Dense(300, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(100, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])
    return model
# FFNN model
ffnn = get_ffnn()

# Model summary
ffnn.summary()
X_valid.shape

# Compile the model
ffnn.compile(optimizer='sgd',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
# Fitting the data
history = ffnn.fit(X_train, y_train, epochs=30, 
                  validation_data=(X_valid, y_valid))
# print(ffnn.layers)

hidden1 = ffnn.layers[1]
print(hidden1.name)
print(ffnn.get_layer('dense_15') is hidden1)

weights, biases = hidden1.get_weights()

print(weights.shape)

print(f"Biases : {biases}")


