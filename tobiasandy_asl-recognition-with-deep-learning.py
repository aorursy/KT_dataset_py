import numpy as np

np.random.seed(5) 

import tensorflow as tf

tf.set_random_seed(2)

import matplotlib.pyplot as plt

%matplotlib inline

import os

import cv2



train_dir = "../input/asl-alphabet/asl_alphabet_train/asl_alphabet_train"

eval_dir = "../input/asl-alphabet-test/asl-alphabet-test"
#Helper function to load images from given directories

def load_images(directory):

    images = []

    labels = []

    for idx, label in enumerate(uniq_labels):

        for file in os.listdir(directory + "/" + label):

            filepath = directory + "/" + label + "/" + file

            image = cv2.resize(cv2.imread(filepath), (64, 64))

            images.append(image)

            labels.append(idx)

    images = np.array(images)

    labels = np.array(labels)

    return(images, labels)
import keras



uniq_labels = sorted(os.listdir(train_dir))

images, labels = load_images(directory = train_dir)



if uniq_labels == sorted(os.listdir(eval_dir)):

    X_eval, y_eval = load_images(directory = eval_dir)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size = 0.1, stratify = labels)



n = len(uniq_labels)

train_n = len(X_train)

test_n = len(X_test)



print("Total number of symbols: ", n)

print("Number of training images: " , train_n)

print("Number of testing images: ", test_n)



eval_n = len(X_eval)

print("Number of evaluation images: ", eval_n)
#Helper function to print images

def print_images(image_list):

    n = int(len(image_list) / len(uniq_labels))

    cols = 8

    rows = 4

    fig = plt.figure(figsize = (24, 12))



    for i in range(len(uniq_labels)):

        ax = plt.subplot(rows, cols, i + 1)

        plt.imshow(image_list[int(n*i)])

        plt.title(uniq_labels[i])

        ax.title.set_fontsize(20)

        ax.axis('off')

    plt.show()
y_train_in = y_train.argsort()

y_train = y_train[y_train_in]

X_train = X_train[y_train_in]



print("Training Images: ")

print_images(image_list = X_train)
y_test_in = y_test.argsort()

y_test = y_test[y_test_in]

X_test = X_test[y_test_in]



print("Testing images: ")

print_images(image_list = X_test)
print("Evaluation images: ")

print_images(image_list = X_eval)
y_train = keras.utils.to_categorical(y_train)

y_test = keras.utils.to_categorical(y_test)

y_eval = keras.utils.to_categorical(y_eval)
print(y_train[0])

print(len(y_train[0]))
X_train = X_train.astype('float32')/255.0

X_test = X_test.astype('float32')/255.0

X_eval = X_eval.astype('float32')/255.0
from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Conv2D, Dense, Dropout, Flatten

from keras.layers import Flatten, Dense

from keras.models import Sequential



model = Sequential()

model.add(Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation = 'relu', 

                 input_shape = (64, 64, 3)))

model.add(Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Dropout(0.5))

model.add(Conv2D(filters = 64 , kernel_size = 3, padding = 'same', activation = 'relu'))

model.add(Conv2D(filters = 128 , kernel_size = 3, padding = 'same', activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Dropout(0.5))

model.add(Conv2D(filters = 128 , kernel_size = 3, padding = 'same', activation = 'relu'))

model.add(Conv2D(filters = 256 , kernel_size = 3, padding = 'same', activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Dropout(0.5))

model.add(Conv2D(filters = 256 , kernel_size = 3, padding = 'same', activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(29, activation='softmax'))



model.summary()
model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
hist = model.fit(X_train, y_train, epochs = 5, batch_size = 64)
score = model.evaluate(x = X_test, y = y_test, verbose = 0)

print('Accuracy for test images:', round(score[1]*100, 3), '%')

score = model.evaluate(x = X_eval, y = y_eval, verbose = 0)

print('Accuracy for evaluation images:', round(score[1]*100, 3), '%')
#Helper function to plot confusion matrix

def plot_confusion_matrix(y, y_pred):

    y = np.argmax(y, axis = 1)

    y_pred = np.argmax(y_pred, axis = 1)

    cm = confusion_matrix(y, y_pred)

    plt.figure(figsize = (24, 20))

    ax = plt.subplot()

    plt.imshow(cm, interpolation = 'nearest', cmap = plt.cm.Purples)

    plt.colorbar()

    plt.title("Confusion Matrix")

    tick_marks = np.arange(len(uniq_labels))

    plt.xticks(tick_marks, uniq_labels, rotation=45)

    plt.yticks(tick_marks, uniq_labels)

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    ax.title.set_fontsize(20)

    ax.xaxis.label.set_fontsize(16)

    ax.yaxis.label.set_fontsize(16)

    limit = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment = "center",color = "white" if cm[i, j] > limit else "black")

    plt.show()
from sklearn.metrics import confusion_matrix

import itertools



y_test_pred = model.predict(X_test, batch_size = 64, verbose = 0)

plot_confusion_matrix(y_test, y_test_pred)
y_eval_pred = model.predict(X_eval, batch_size = 64, verbose = 0)

plot_confusion_matrix(y_eval, y_eval_pred)