import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))
training_dataset = pd.read_csv('../input/fashion-mnist_train.csv')

testing_dataset = pd.read_csv('../input/fashion-mnist_test.csv')

print(training_dataset.head())

print(testing_dataset.head())
train_labels = training_dataset['label'].values

test_labels = testing_dataset['label'].values



"""

train_labels = pd.get_dummies(training_labels).values

test_labels = pd.get_dummies(testing_labels).values

"""



print(train_labels.shape)

print(test_labels.shape)
train_features = training_dataset.drop(['label'], axis=1).values

test_features = testing_dataset.drop(['label'], axis=1).values



print(train_features.shape)

print(test_features.shape)
train_images = train_features.reshape(train_features.shape[0], 28, 28, 1)

test_images = test_features.reshape(test_features.shape[0], 28, 28, 1)



print(train_images.shape)

print(test_images.shape)
import keras

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.10)

#TODO : Change image augmentation techniques used

test_datagen= ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow(

    train_images,

    train_labels,

    batch_size=400,

    subset="training"

)



val_generator = train_datagen.flow(

    train_images,

    train_labels,

    batch_size=400,

    subset="validation"

)



test_generator = test_datagen.flow(

    test_images,

    test_labels,

    batch_size=1,

    shuffle=True

)
model = Sequential()

model.add(Conv2D(filters=6, kernel_size=(5,5), activation='relu', input_shape=(28,28,1)))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=16, kernel_size=(5,5), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(units=120, activation='relu'))

model.add(Dropout(rate=0.5))

model.add(Dense(units=84, activation='relu'))

model.add(Dropout(rate=0.5))

model.add(Dense(units=10, activation='softmax'))

model.summary()
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['acc'])

history = model.fit_generator(

    train_generator,

    epochs=100,

    steps_per_epoch=135, 

    validation_data=val_generator,

    validation_steps=10

)
acc, val_acc = history.history['acc'], history.history['val_acc']

loss, val_loss = history.history['loss'], history.history['val_loss']

epochs = range(len(acc))

import matplotlib.pyplot as plt



plt.plot(epochs, acc, label='Training accuracy')

plt.plot(epochs, val_acc, label='Validation accuracy')

plt.legend()

plt.title('Accuracy Curve')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.show()



plt.plot(epochs, loss, label='Training loss')

plt.plot(epochs, val_loss, label='Validation loss')

plt.legend()

plt.title('Loss Curve')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.show()
test_loss, test_acc = model.evaluate_generator(

    test_generator,

    steps=len(test_generator)

)



print("Test accuracy : {}".format(test_acc * 100))

print("Test loss/error : {}".format(test_loss))