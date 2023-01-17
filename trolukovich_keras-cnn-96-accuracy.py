# Imports

import numpy as np

import pandas as pd

import cv2 as cv

import matplotlib.pyplot as plt

import seaborn as sns

import os

import warnings

import random



from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization

from keras.utils.np_utils import to_categorical

from keras.callbacks import ModelCheckpoint

from keras.optimizers import Adam



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



warnings.filterwarnings('ignore')
# Path to classes

path = '../input/cell-images-for-detecting-malaria/cell_images/cell_images'



CLASSES = sorted([i for i in os.listdir(path)], reverse = True)

print(CLASSES)



# Defining image size and number of dimensions for resized images

IMG_SIZE = (64, 64)

N_DIMS = 3
# Plot random images of each class

rows, cols = (2, 7)



for c in CLASSES:

    print(f'{c} cells:')

    path_to_folder = os.path.join(path, c)

    fig = plt.figure(figsize = (18, 6))

    

    for i in range(rows*cols):

        random_image = random.choice(os.listdir(path_to_folder))

        path_to_image = os.path.join(path_to_folder, random_image)        

        image = cv.imread(path_to_image)

        image = cv.resize(image, (IMG_SIZE[0], IMG_SIZE[1]))

        

        fig.add_subplot(rows, cols, i+1)

        plt.imshow(image[:, :, ::-1])

        plt.axis('off')

        plt.title(image.shape)

    plt.show()
# Creating datasets

X = []

Y = []



for c in CLASSES:

    path_to_images = os.path.join(path, c)

    label = CLASSES.index(c)

    print(f'"{c}" class label - {label}')    

    

    for i in os.listdir(path_to_images):        

        try:

            image = cv.imread(os.path.join(path_to_images, i))

            image = cv.resize(image, (IMG_SIZE[0], IMG_SIZE[1]))



            X.append(image)

            Y.append(label)

        except:

            print(f'Can\'t read {c, i}')
# Conver images list to numpy array, reshape and scale it

X = np.array(X).reshape(-1, IMG_SIZE[0], IMG_SIZE[1], N_DIMS)

X = X / 255.0



# Split data to training and test datasets

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, shuffle = True, stratify = Y, random_state = 666)



# Binarizing labels

Y_train = to_categorical(Y_train, num_classes = 2)
batch_size = 32



# Creating model

model = Sequential()

model.add(Conv2D(32, 3, input_shape = (IMG_SIZE[0], IMG_SIZE[1], N_DIMS), padding = 'same', activation = 'relu', kernel_initializer = 'he_normal'))

model.add(Conv2D(32, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal'))

model.add(MaxPooling2D(2))



model.add(Conv2D(64, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal'))

model.add(Conv2D(64, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal'))

model.add(MaxPooling2D(2))



model.add(Conv2D(128, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal'))

model.add(Conv2D(128, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal'))

model.add(MaxPooling2D(2))



model.add(Conv2D(256, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal'))

model.add(Conv2D(256, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal'))

model.add(MaxPooling2D(2))



model.add(Flatten())

model.add(Dense(128, activation = 'relu'))

model.add(Dense(100, activation = 'relu'))

model.add(Dense(2, activation = 'softmax'))



# model.summary()

checkpoint = ModelCheckpoint('../working/best_model.hdf5', verbose = 1, monitor = 'val_loss', save_best_only = True)

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = 10, validation_split = 0.2, callbacks = [checkpoint])
# Plot learning curves

fig = plt.figure(figsize = (10, 6))

plt.plot(history.history['accuracy'], label = 'acc')

plt.plot(history.history['val_accuracy'], label = 'val_acc')

plt.plot(history.history['loss'], label = 'loss')

plt.plot(history.history['val_loss'], label = 'val_loss')

plt.grid()

plt.legend()
# Load best model

model.load_weights('../working/best_model.hdf5')



# Making predictions

preds = np.argmax(model.predict(X_test), axis = 1)



# Calculating and printing accuracy

print(f'Accuracy: {round(accuracy_score(Y_test, preds), 4) * 100}%')



# Plotting confusion matrix

confusion = confusion_matrix(Y_test, preds)

sns.heatmap(confusion, annot = True, square = True, fmt = 'd', xticklabels = CLASSES, yticklabels = CLASSES)

plt.show()