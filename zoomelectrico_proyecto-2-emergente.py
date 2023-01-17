import numpy as np

import pandas as pd

import keras

import tensorflow as tf

import tensorflow.keras.backend as K

from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import layers, models

from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization

from keras import regularizers

%matplotlib inline

import matplotlib.pyplot as plt

from matplotlib.pyplot import imshow

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, roc_auc_score

import itertools

import os

import time

print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
def clean_inputs(train, test, img_shape = (-1,28,28,1), num_classes = 10):

    t_X = train.drop("label", axis=1)

    t_Y = train["label"]

    t_X = t_X / 255

    test_x = test.values / 255

    

    t_X = np.reshape(t_X.values, img_shape)

    test_x = np.reshape(test_x, img_shape)

    

    t_Y = keras.utils.to_categorical(t_Y, num_classes = num_classes)

    train_x, dev_x, train_y, dev_y = train_test_split(t_X, t_Y, test_size = 0.15, random_state = 0)

    

    return train_x, train_y, dev_x, dev_y, test_x
train_x, train_y, dev_x, dev_y, test_x = clean_inputs(train, test)
def showRandImages(): 

    # show random images from train

    random_index = np.random.randint(0,train_x.shape[0])

    plt.imshow(train_x[random_index][:,:,0])

    print(train_y[random_index])
showRandImages()
precisiones_globales=[]

def plot_model(history):

    # Plot Accuracy

    plt.plot(history.history['acc'])

    plt.plot(history.history['val_acc'])

    plt.title('model accuracy')

    plt.ylabel('accuracy')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.show()

    # Plot Loss

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('model loss')

    plt.ylabel('loss')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.show()
# Propuesta de Modelo A Basado en [1]

def model(inp_shape):

    # (32 conv -> 28x6) -> (pool 14x6) -> (conv -> 10X16) -> (pool -> 5X16) - -> flatte -> FC120 -> FC84 -> softmax10

    X = Input(inp_shape, name='input')

    # 32x32x1 -> conv -> 28x28x6 -> pool -> 14x14x6

    A = Conv2D(6, (7, 7), strides=(1, 1), padding='Same', activation='relu', name='C1')(X)

    A = MaxPooling2D(pool_size=2, padding='valid')(A)

    # 14x14x6 -> conv -> 10x10x16 -> pool -> 5x5x16

    A = Conv2D(16, (5, 5), strides=(1, 1), padding='Same', activation='relu', name='C2')(A)

    A = MaxPooling2D(pool_size=2, padding='valid')(A)

    # flatten 5*5*16 = 400

    A = Flatten()(A)

    # normalization -> FC 120 -> FC 84 -> softmax 10

    A = BatchNormalization()(A)

    A = Dense(120, activation='relu', kernel_regularizer=regularizers.l2(0.01), name='FC1')(A)

    A = Dense(84, activation='relu', kernel_regularizer=regularizers.l2(0.01), name='FC2')(A)

    A = Dense(10, activation='softmax', name='Final')(A)

    model = Model(inputs=X, outputs=A, name='LeNet')

    return model
datagen = ImageDataGenerator(       

        rotation_range=10,  

        zoom_range = 0.1, 

        width_shift_range=0.1,  

        height_shift_range=0.1)        

datagen.fit(train_x)
# Adding pad to the image

train_x_pad = np.pad(train_x, ((0,0), (2,2), (2,2), (0,0)), mode='constant', constant_values=0).astype(float)

dev_x_pad = np.pad(dev_x, ((0,0), (2,2), (2,2), (0,0)), mode='constant', constant_values=0).astype(float)

test_x_pad = np.pad(test_x, ((0,0), (2,2), (2,2), (0,0)), mode='constant', constant_values=0).astype(float)

# Learning Rate Decay

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1,factor=0.5, min_lr=0.00001)

# model

model = model(train_x_pad.shape[1:])

model.summary()

model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'] )

# model history

history = model.fit_generator(datagen.flow(train_x_pad, train_y, batch_size=32), validation_data=(dev_x_pad, dev_y), steps_per_epoch=len(train_x_pad)//32, epochs=25, callbacks=[learning_rate_reduction])
plot_model(history)
# Modelo basado en [2]

def model2(num_classes = 10):

    model = Sequential()

    # [conv2D -> conv2D -> MaxPooling2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Dense (Out)

    model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'same', activation ='relu', input_shape = (28,28,1)))

    model.add(BatchNormalization())

    

    model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'same', activation ='relu'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

 

    model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', activation ='relu'))

    model.add(BatchNormalization())

   

    model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', activation ='relu'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

    

    model.add(Flatten())

    model.add(Dense(256, activation = "relu"))

    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation = "softmax"))

    

    return model;
start = time.time()

#del model2

model2 = model2(10)



# Learning Rate Decay

learning_rate_reduction2 = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1,factor=0.5, min_lr=0.00001)



model2.summary()

model2.compile('adam', 'categorical_crossentropy', metrics=['accuracy'] )

history2 = model2.fit_generator(datagen.flow(train_x, train_y, batch_size=32), validation_data=(dev_x, dev_y), steps_per_epoch=len(train_x)//32, epochs=25, callbacks=[learning_rate_reduction2])



timeRecord = time.time() - start

print("--- %s seconds ---" % (timeRecord))
plot_model(history2)
# Modelo basado en [3]

def model3(num_classes = 10):

    model = Sequential()

    # [conv2D -> MaxPooling2D -> conv2D ]*3 -> Flatten -> Dense -> Dense (Out)

    model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'same', activation ='relu', input_shape = (28,28,1)))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    

    model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', activation ='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

 

    model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', activation ='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    

    model.add(Flatten())

    model.add(Dense(256, activation = "relu"))

    model.add(Dense(num_classes, activation = "softmax"))

    

    return model;
start = time.time()

model3 = model3(10)



# Learning Rate Decay

learning_rate_reduction3 = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1,factor=0.5, min_lr=0.00001)



model3.summary()

model3.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'] )

history3 = model3.fit_generator(datagen.flow(train_x, train_y, batch_size=64), 

                                validation_data=(dev_x, dev_y), steps_per_epoch=len(train_x)//64, epochs=25, 

                                callbacks=[learning_rate_reduction3])



timeRecord = time.time() - start

print("--- %s seconds ---" % (timeRecord))
plot_model(history3)
prediction = model2.predict(test_x)

prediction = np.argmax(prediction, axis=1)

prediction = pd.Series(prediction, name="Label")

submission = pd.concat([pd.Series(range(1,28001), name = "ImageId"), prediction],axis = 1)

submission.to_csv('mnist-submission.csv', index = False)

print(submission)