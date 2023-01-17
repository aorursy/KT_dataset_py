from tensorflow import keras

from tensorflow.keras.datasets import fashion_mnist

from tensorflow.keras.layers import Dense

from tensorflow.keras.models import Sequential

import numpy as np

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Conv2D, Flatten, Dense, AveragePooling2D, GlobalAveragePooling2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications.resnet import ResNet50

import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf
labels = pd.read_csv('/kaggle/input/appa-real-face-cropped/labels.csv')

train_datagen = ImageDataGenerator(rescale=1./255)
train_gen_flow = train_datagen.flow_from_dataframe(

        dataframe=labels,

        directory='/kaggle/input/appa-real-face-cropped/final_files/final_files/',

        x_col='file_name',

        y_col='real_age',

        target_size=(224, 224),

        batch_size=32,

        class_mode='raw',

        seed=12345

)
features, target = next(train_gen_flow)



fig = plt.figure(figsize=(10,10))

for i in range(10):

    fig.add_subplot(4, 4, i+1)

    plt.imshow(features[i])

    plt.xticks([])

    plt.yticks([])

    plt.tight_layout()
labels.hist(bins=100,density=True)
# load train

train_datagen = ImageDataGenerator(validation_split=0.25, rescale=1./255,shear_range=0.2,

                                    horizontal_flip=True,)

#добавили аугментацию.



train_data = train_datagen.flow_from_dataframe(

        dataframe=labels,

        directory='/kaggle/input/appa-real-face-cropped/final_files/final_files/',

        x_col='file_name',

        y_col='real_age',

        target_size=(150, 150),

        batch_size=32,

        class_mode='raw',

        seed=12345,

        subset='training')
#load test

test_datagen = ImageDataGenerator(validation_split=0.25, rescale=1./255)

test_data = test_datagen.flow_from_dataframe(

        dataframe=labels,

        directory='/kaggle/input/appa-real-face-cropped/final_files/final_files/',

        x_col='file_name',

        y_col='real_age',

        target_size=(150, 150),

        batch_size=32,

        class_mode='raw',

        seed=12345,

        subset='validation')
#create model



optimizer= Adam() 

backbone = ResNet50(input_shape=(150,150,3),weights='imagenet', include_top=False)

model = Sequential()

model.add(backbone)

model.add(GlobalAveragePooling2D())

model.add(Dense(1, activation='relu'))

model.compile(loss="mean_squared_error",optimizer=optimizer, metrics=["mean_absolute_error"])
def train_model(model, train_data, test_data,batch_size=None,epochs=10, steps_per_epoch =None, validation_steps=None):

    my_callbacks = [ tf.keras.callbacks.EarlyStopping(patience=35)] #ранняя остановка

    if steps_per_epoch is None:

        steps_per_epoch = len(train_data)

    if validation_steps is None:

        validation_steps = len(test_data)

    model.fit(train_data,

          validation_data=test_data,

          steps_per_epoch=steps_per_epoch,

          validation_steps=validation_steps,

          verbose=1, epochs=epochs,  callbacks=my_callbacks)

    return model
trained_model = train_model(model, train_data, test_data,batch_size=None,epochs=300, steps_per_epoch =None, validation_steps=None)
#Epoch 35/150 adam tandart, +augmentation

#178/178 [==============================] - 98s 548ms/step - loss: 10.7784 - mean_absolute_error: 2.5221 - val_loss: 80.2944 - val_mean_absolute_error: 6.6870
trained_model.save("model.h5")