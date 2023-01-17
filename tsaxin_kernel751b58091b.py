import numpy as np

np.random.seed(5) 

import tensorflow as tf

tf.set_random_seed(2)

import matplotlib.pyplot as plt

%matplotlib inline

import os

import cv2



train_dir = "../input/asl_alphabet_train/asl_alphabet_train"

eval_dir = "../input/asl_alphabet_test/asl_alphabet_test"
#Helper function to load images from given directories

def load_images(directory):

    images = []

    labels = []

    for idx, label in enumerate(uniq_labels):

        for file in os.listdir(directory + "/" + label):

            filepath = directory + "/" + label + "/" + file

            image=cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)

            image = cv2.resize((image), (100, 100))

            image=np.reshape(image,[100,100,1])

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



#print("Training Images: ")

#print_images(image_list = X_train)
y_test_in = y_test.argsort()

y_test = y_test[y_test_in]

X_test = X_test[y_test_in]



#print("Testing images: ")

#print_images(image_list = X_test)
y_train = keras.utils.to_categorical(y_train)

y_test = keras.utils.to_categorical(y_test)
print(y_train[0])

print(len(y_train[0]))
X_train = X_train.astype('float32')/255.0

X_test = X_test.astype('float32')/255.0
from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Conv2D, Dense, Dropout, Flatten

from keras.layers import Flatten, Dense

from keras.models import Sequential



model = Sequential()

model.add(Conv2D(filters = 64, kernel_size = 5, padding = 'same', activation = 'relu', 

                 input_shape = (100, 100, 1)))

model.add(Conv2D(filters = 64, kernel_size = 5, padding = 'same', activation = 'relu'))

model.add(MaxPooling2D(pool_size = (4, 4)))

model.add(Dropout(0.5))

model.add(Conv2D(filters = 128 , kernel_size = 5, padding = 'same', activation = 'relu'))

model.add(Conv2D(filters = 128 , kernel_size = 5, padding = 'same', activation = 'relu'))

model.add(MaxPooling2D(pool_size = (4, 4)))

model.add(Dropout(0.5))

model.add(Conv2D(filters = 256 , kernel_size = 5, padding = 'same', activation = 'relu'))

model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(29, activation='softmax'))
model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
hist = model.fit(X_train, y_train, epochs = 10, batch_size = 64)
plt.plot(hist.history['acc'])

plt.plot(hist.history['loss'])

plt.title('Accuracy plot')

plt.xlabel('epochs')

plt.ylabel('Range')

plt.legend(['accuracy', 'loss'])

plt.show()
model_json = model.to_json()

with open("model.json", "w") as json_file:

    json_file.write(model_json)

model.save("model_Grayscale.h5")

print("Saved model to disk")
from keras.models import load_model



model = load_model('model_Grayscale.h5')



model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])



filepath="../input/asl_alphabet_test/asl_alphabet_test/B_test.jpg"

img = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)

img = cv2.resize(img, (100, 100))



img = np.reshape(img,[1,100,100,1])



classes = model.predict_classes(img)



print(classes)