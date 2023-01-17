import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import keras
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
fig = plt.figure(figsize=(8,8))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.tight_layout()
    plt.imshow(X_train[i], cmap = 'gray', interpolation=None)
    plt.title("Digit: {}".format(y_train[i]))
    plt.xticks([])
    plt.yticks([])
from keras.utils import np_utils
y_cat_train = np_utils.to_categorical(y_train)
y_cat_test = np_utils.to_categorical(y_test)
X_train = X_train / 255
X_test = X_test / 255
X_train.shape
X_test.shape
X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)
X_train.shape
X_test.shape
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Activation
model = Sequential()
model.add(Conv2D(32, kernel_size=(4, 4), input_shape= (28,28,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(X_train, y_cat_train, batch_size=5000, epochs=50)
model.evaluate(X_test, y_cat_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
predictions = model.predict_classes(X_test)
y_cat_test
y_test
predictions
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
acc = accuracy_score(y_test, predictions)
print("Test Accuracy is :",acc * 100, '%')