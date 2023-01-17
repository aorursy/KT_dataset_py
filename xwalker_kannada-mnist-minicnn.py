import numpy as np 

import pandas as pd 

from glob import glob

import matplotlib.pyplot as plt

from skimage.io import imread

from skimage.transform import resize

%matplotlib inline

from sklearn.model_selection import train_test_split

import keras

from keras.preprocessing.image import ImageDataGenerator

from keras.utils import print_summary

from keras.models import Sequential

from keras.layers import (  Dense, 

                            Conv2D,

                            MaxPooling2D,

                            MaxPool2D,

                            Dropout,

                            BatchNormalization,

                            Flatten

                         )

from keras.optimizers import Adam, RMSprop

from keras.utils import print_summary

from keras.datasets import mnist

from keras.utils.np_utils import to_categorical
def plot(img):

    plt.imshow(img);

    plt.show();
def history(History):

    

    fig, ax = plt.subplots(1, 2, figsize=(20, 5))

    

    ax[0].plot(History.history['loss'])

    ax[0].plot(History.history['val_loss'])

    ax[0].legend(['Training loss', 'Validation Loss'],fontsize=18)

    ax[0].set_xlabel('Epochs ',fontsize=16)

    ax[0].set_ylabel('Loss',fontsize=16)

    ax[0].set_title('Training loss x Validation Loss',fontsize=16)



    ax[1].plot(History.history['accuracy'])

    ax[1].plot(History.history['val_accuracy'])

    ax[1].legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)

    ax[1].set_xlabel('Epochs ',fontsize=16)

    ax[1].set_ylabel('Accuracy',fontsize=16)

    ax[1].set_title('Training Accuracy x Validation Accuracy',fontsize=16)

train_id = pd.read_csv('../input/Kannada-MNIST/train.csv')

test_id = pd.read_csv('../input/Kannada-MNIST/test.csv')

dig_id = pd.read_csv('../input/Kannada-MNIST/Dig-MNIST.csv')
x, y, z = 28, 28, 1

qtd_classes = 10



images = []

images_labels = []



valid = []

valid_labels = []



for count, (index, row) in enumerate(train_id.iterrows()):

    images.append(row.values[1:].reshape(x, y, z))

    images_labels.append(row.values[:1])

    

for count, (index, row) in enumerate(dig_id.iterrows()):

    valid.append(row.values[1:].reshape(x, y, z))

    valid_labels.append(row.values[:1])
TESTER = []

for count, (index, row) in enumerate(test_id.iterrows()):

    TESTER.append(row.values[1:].reshape(x, y, z))
SUBMIT =  np.asarray(TESTER) / 255.

TEST   =  np.asarray(valid) / 255.

TRAIN  =  np.asarray(images) / 255.



TRAIN_labels = to_categorical(np.asarray(images_labels), qtd_classes)

TEST_labels  = to_categorical(np.asarray(valid_labels), qtd_classes)
train = np.concatenate([TEST, TRAIN])

labels = np.concatenate([TRAIN_labels, TEST_labels])
plt.figure(figsize = (15, 5))

for i in range(0,10):

    plt.subplot(2,5,i + 1)

    plt.xticks([])

    plt.yticks([])

    plt.title(str(np.argmax(labels[i])))

    plt.imshow(train[i,:,:,0], cmap="inferno")
x_train, x_test, y_train, y_test = train_test_split(train, labels, test_size = 0.3)
NUM_CLASSES = 10

EPOCHS = 15

BATCH_SIZE = 64

inputShape = (x, y, z)
model = Sequential()

model.add(Conv2D(16, kernel_size = 3, padding="same", activation='relu', input_shape = inputShape))



model.add(Conv2D(32, kernel_size = 3, padding="same", activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, kernel_size = 3, padding="same", activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(128, kernel_size = 3, padding="same", activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(256, kernel_size = 3, padding="same", activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(NUM_CLASSES, activation='softmax'))

    

optmize = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

model.compile(loss = keras.losses.categorical_crossentropy, optimizer = optmize,  metrics=['accuracy'])

History = model.fit(x_train, y_train,

          batch_size = BATCH_SIZE,

          epochs = EPOCHS,

          verbose = 1,

          validation_data=(x_test, y_test))
loss, accu = model.evaluate(x_test, y_test)

print("%s: %.2f%%" % ('Accuracy...', accu))

print("%s: %.2f" % ('Loss...', loss))
history(History)
History = model.fit(train, labels,

          batch_size = BATCH_SIZE,

          epochs = 100,

          verbose = 1)
results = model.predict(SUBMIT)

results = np.argmax(results,axis = 1)

data_out = pd.DataFrame({'id': range(len(SUBMIT)), 'label': results})
plt.figure(figsize = (15, 5))

for i in range(0,10):

    plt.subplot(2,5,i + 1)

    plt.xticks([])

    plt.yticks([])

    plt.title(str(results[i]))

    plt.imshow(TESTER[i][:,:,0], cmap="inferno")
data_out.head()
data_out.to_csv('submission.csv', index = None)