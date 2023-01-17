import os 

import cv2

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow as tf

import keras

from keras.models import Sequential,Model

from keras.layers import Dense,Flatten,Dropout,BatchNormalization,Conv2D,MaxPool2D

from keras.preprocessing.image import ImageDataGenerator

from tqdm import tqdm

from sklearn.model_selection import train_test_split

from  skimage.transform import resize

from keras.utils import to_categorical

from keras.applications.inception_v3 import InceptionV3

from keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.optimizers import RMSprop, Adam, SGD

import pickle
training_dir="../input/indian-sign-language-dataset/data"

content=sorted(os.listdir(training_dir))

print(content)

len(content)
data_generator = ImageDataGenerator(

    samplewise_center=True, 

    samplewise_std_normalization=True,

    brightness_range=[0.8, 1.0],

    zoom_range=[1.0, 1.2],

    validation_split=0.1

)



train_generator = data_generator.flow_from_directory(training_dir, target_size=(200,200), shuffle=True, seed=13,

                                                     class_mode='categorical', batch_size=64, subset="training")



validation_generator = data_generator.flow_from_directory(training_dir, target_size=(200, 200), shuffle=True, seed=13,

                                                     class_mode='categorical', batch_size=64, subset="validation")
WEIGHTS_FILE = './inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'



inception_v3_model = keras.applications.inception_v3.InceptionV3(

    input_shape = (200, 200, 3), 

    include_top = False, 

    weights = 'imagenet'

)



inception_v3_model.summary()
inception_output_layer = inception_v3_model.get_layer('mixed7')

print('Inception model output shape:', inception_output_layer.output_shape)



inception_output = inception_v3_model.output
from tensorflow.keras import layers

x = layers.GlobalAveragePooling2D()(inception_output)

x = layers.Dense(1024, activation='relu')(x)                  

x = layers.Dense(35, activation='softmax')(x)           



model = Model(inception_v3_model.input, x) 



model.compile(

    optimizer=SGD(lr=0.0001, momentum=0.9),

    loss='categorical_crossentropy',

    metrics=['acc']

)

for layer in model.layers[:249]:

    layer.trainable = False

for layer in model.layers[249:]:

    layer.trainable = True
LOSS_THRESHOLD = 0.2

ACCURACY_THRESHOLD = 0.979



class ModelCallback(tf.keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs={}):

    if logs.get('val_loss') <= LOSS_THRESHOLD and logs.get('val_acc') >= ACCURACY_THRESHOLD:

      print("\nReached", ACCURACY_THRESHOLD * 100, "accuracy, Stopping!")

      self.model.stop_training = True



callback = ModelCallback()

history = model.fit_generator(

    train_generator,

    validation_data=validation_generator,

    steps_per_epoch=200,

    validation_steps=50,

    epochs=50,

    callbacks=[callback]

)

model.save('transferlearning2.h5')
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







plt.show()
plt.plot(epochs, loss, 'r', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validationloss')

plt.title('Training and validation loss')

plt.legend(loc=0)

plt.figure()