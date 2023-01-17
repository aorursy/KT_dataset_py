import os

print(os.listdir("../input"))

import numpy as np

import matplotlib.pyplot as plt

from glob import glob

import PIL

from PIL import Image

from keras.preprocessing.image import ImageDataGenerator
image="../input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/person1011_bacteria_2942.jpeg"

PIL.Image.open(image)
image="../input/chest-xray-pneumonia/chest_xray/train/NORMAL/IM-0151-0001.jpeg"

PIL.Image.open(image)
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
training_dir="../input/chest-xray-pneumonia/chest_xray/train/"

training_generator=ImageDataGenerator(rescale=1/255,featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.2, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip = False,  # randomly flip images

        vertical_flip=False)

train_generator=training_generator.flow_from_directory(training_dir,target_size=(200,200),batch_size=4,class_mode='binary')
validation_dir="../input/chest-xray-pneumonia/chest_xray/val/"

validation_generator=ImageDataGenerator(rescale=1/255)

val_generator=validation_generator.flow_from_directory(validation_dir,target_size=(200,200),batch_size=4,class_mode='binary')
test_dir="../input/chest-xray-pneumonia/chest_xray/test/"

test_generator=ImageDataGenerator(rescale=1/255)

test_generator=test_generator.flow_from_directory(test_dir,target_size=(200,200),batch_size=16,class_mode='binary')
model=tf.keras.Sequential([

    tf.keras.layers.Conv2D(32,(3,3),input_shape=(200,200,3),activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    

    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.2),

    

    tf.keras.layers.Conv2D(128,(3,3),activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.2),

    

    tf.keras.layers.Conv2D(256,(3,3),activation='relu'), 

    tf.keras.layers.MaxPooling2D(2,2),

    

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(256,activation='relu'),

    tf.keras.layers.Dense(1,activation='sigmoid')

    

])
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),loss='binary_crossentropy',metrics=['acc'])
history = model.fit_generator(train_generator,

            validation_data = val_generator,

            

            epochs = 30,

            

            verbose = 1)
acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'r', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend(loc=0)

plt.figure()
print("Loss of the model is - " , model.evaluate(test_generator)[0]*100 , "%")

print("Accuracy of the model is - " , model.evaluate(test_generator)[1]*100 , "%")