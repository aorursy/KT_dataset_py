



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import seaborn as sns

import matplotlib.pyplot as plt

from tensorflow import keras



from sklearn.model_selection import  train_test_split

from keras.datasets import mnist

from keras import models

from keras import layers

from keras.utils import to_categorical



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv')

sample_sub = pd.read_csv('../input/digit-recognizer/sample_submission.csv')
print(train.shape, test.shape, sample_sub.shape)
y_train = train["label"]

y_train.shape
y_train.value_counts()
g = sns.countplot(y_train)
# Drop 'label' column

X_train = train.drop(labels = ["label"],axis = 1)

X_train.shape
del train
# Normalize the values

X_train = X_train/255.0

test = test/255.0

# Reshape to be examples of 28x28 pixel images

X_train = X_train.values.reshape(-1, 28, 28, 1)

test = test.values.reshape(-1, 28, 28, 1)



X_train.shape, test.shape
# Convert to one-hot encoding

y_train = to_categorical(y_train, num_classes = 10)

y_train[9]
# Split into training and validation sets

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state = 2)
model = keras.models.Sequential([

    keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (28, 28, 1)),

    keras.layers.MaxPooling2D(2, 2),

    keras.layers.Dropout(0.5),

    keras.layers.Conv2D(64, (3, 3), activation = 'relu'),

    keras.layers.MaxPooling2D(2, 2),

    keras.layers.Dropout(0.5),

    keras.layers.Conv2D(64, (3, 3), activation = 'relu'),

    keras.layers.Flatten(),

    keras.layers.Dense(128, activation = 'relu'),

    keras.layers.Dropout(0.5),

    keras.layers.Dense(10, activation = 'softmax')

])



model.summary()
# Use categorical_crossentropy as loss because problem is multiclass classification problem

model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['acc'])
history = model.fit(X_train, y_train, epochs = 20, batch_size = 128, validation_data = (X_val, y_val), verbose = 2)
loss = history.history['loss']

val_loss = history.history['val_loss']

acc = history.history['acc']

val_acc = history.history['val_acc']
epochs = range(1, 21)



plt.plot(epochs, loss, 'ko', label = 'Training Loss')

plt.plot(epochs, val_loss, 'k', label = 'Validation Loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.title('Training and Validation Loss')

plt.legend()
plt.plot(epochs, acc, 'yo', label = 'Training Accuracy')

plt.plot(epochs, val_acc, 'y', label = 'Validation Accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.title('Training and Validation Accuracy')

plt.legend()
results = model.predict(test)
results = np.argmax(results, axis = 1)

results = pd.Series(results, name = 'Label')
submission = pd.concat([pd.Series(range(1, 28001), name = 'ImageId'), results], axis = 1)

submission.to_csv("MNIST_Dataset_Submissions.csv", index = False)
from IPython.display import FileLink

FileLink('MNIST_Dataset_Submissions.csv')