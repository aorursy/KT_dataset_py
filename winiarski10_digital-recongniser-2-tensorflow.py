import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
%matplotlib inline

image_file = pd.read_csv("../input/train.csv")
image_values = image_file.values
images = image_values[:,1:].reshape(image_file.shape[0], 28, 28)
image_labels = image_values[:,0]

train_images, test_images, train_labels, test_labels = train_test_split(images, image_labels, test_size=0.2)

train_images = train_images / 255.0
test_images = test_images / 255.0
images_1 = image_values[:,1:].reshape(image_file.shape[0], 28, 28, 1)
image_labels_1 = image_values[:,0]

train_images_1, test_images_1, train_labels_1, test_labels_1 = train_test_split(images_1, image_labels_1, test_size=0.2)

train_images_1 = train_images_1 / 255.0
test_images_1 = test_images_1 / 255.0
train_images_1.shape
plt.figure(figsize=[10,10])
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(train_labels[i])

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)
new_model = keras.Sequential([
    keras.layers.Conv2D(12, kernel_size=[3,3], activation=tf.nn.relu, input_shape=(28,28, 1)),
    keras.layers.Conv2D(12, kernel_size=[3,3], activation=tf.nn.relu),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

new_model.compile(optimizer=tf.train.AdamOptimizer(),
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

new_model.fit(train_images_1, train_labels_1, epochs=3)
test_loss, test_acc = model.evaluate(test_images, test_labels)
new_test_loss, new_test_acc = new_model.evaluate(test_images_1, test_labels_1)
print('Test accuracy:', test_acc)
print('New Test accuracy:', new_test_acc)