import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
print(tf.__version__)
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
train_images.shape
test_images.shape
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()
np.unique(train_labels, return_counts=True)
# Normalizing data
train_images = train_images / 255.0
test_images = test_images / 255.0
# How well a human can classify an image
humanLevelPerformance = 0.9

# How bad a human can classify an image
humanLevelError = 0.1
# Input configuration
inputHeight = train_images.shape[1]
inputWeight = train_images.shape[2]
numberChannels = train_images.shape[3]
# Hyperparameters

# 1028 because it is fast and the data is small
batchSize = 1028

# 500 epochs because it is necessary a large number of iteration to get the best results
epochs = 500
AUTOTUNE = tf.data.experimental.AUTOTUNE
# Large size kernel use to large amount of pixels (big images)
# For small images or many elements, small kernels

model = tf.keras.models.Sequential()

# 3x3 filter because the input image is small and I want to capture as many details as posible.
# 32 filters because I follow LeNet-5 recommendation as start point in order to get.
# padding='same' because I want to capture image's borders.
# Activation function='relu' because it is the most recommended.
# MaxPooling2D(2, 2) to shrink convolution layer size and speed training and reduce risk of overfitting.
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(inputHeight, inputWeight, numberChannels)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))

# 3x3 filter because to keep capturing as many details as posible
# 64 filters because more neurons process more information (feature maps).
# MaxPooling2D(2, 2) to shrink convolution layer size and speed training and reduce risk of overfitting
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))

# 3x3 filter because to keep capturing as many details as posible
# 64 filters because more neurons process more information (feature maps).
# MaxPooling2D(2, 2) to shrink convolution layer size and speed training and reduce risk of overfitting
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))

# LeNet-5 recommendations
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10))

model.summary()
# Adam optimizer because it is most recommended
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
# EarlyStopping to capture the best loss
history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batchSize, validation_data=(test_images, test_labels),
                    callbacks = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5))
# Best results
print('Loss:', history.history['loss'][-1])
print('Accuracy:', history.history['accuracy'][-1])
print('Val Loss:', history.history['val_loss'][-1])
print('Val Accuracy:', history.history['val_accuracy'][-1])
avoidableBias = history.history['loss'][-1] - humanLevelError
variance = history.history['val_loss'][-1] - history.history['loss'][-1]

print('Avoidable bias:', avoidableBias)
print('Variance:', variance)

if avoidableBias < variance:
  print('It is necessary to reduce variance')
else:
  print('It is necessary to reduce bias')
model = tf.keras.models.Sequential()

# Add Dropout performed better in CNN layers
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(inputHeight, inputWeight, numberChannels)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Dropout(0.2))

# Add regulation L2 performed better in Fully connected layers
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer = tf.keras.regularizers.L2(0.01)))
model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer = tf.keras.regularizers.L2(0.01)))
model.add(tf.keras.layers.Dense(10))

model.summary()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batchSize, validation_data=(test_images, test_labels),
                    callbacks = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5))
# Best results
print('Loss:', history.history['loss'][-1])
print('Accuracy:', history.history['accuracy'][-1])
print('Val Loss:', history.history['val_loss'][-1])
print('Val Accuracy:', history.history['val_accuracy'][-1])
avoidableBias = history.history['loss'][-1] - humanLevelError
variance = history.history['val_loss'][-1] - history.history['loss'][-1]

print('Avoidable bias:', avoidableBias)
print('Variance:', variance)

if avoidableBias < variance:
  print('It is necessary to reduce variance')
else:
  print('It is necessary to reduce bias')