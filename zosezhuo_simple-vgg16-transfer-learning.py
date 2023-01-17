# Import relevant dependencies.



import numpy as np

import keras

from keras.models import Model, Sequential

from keras.utils import to_categorical

from keras import backend as K

from keras.optimizers import Adam

import os

print(os.listdir('../input'))



train_data = np.load('../input/kuzushiji/kmnist-train-imgs.npz')['arr_0']

test_data = np.load('../input/kuzushiji/kmnist-test-imgs.npz')['arr_0']

train_labels = np.load('../input/kuzushiji/kmnist-train-labels.npz')['arr_0']

test_labels = np.load('../input/kuzushiji/kmnist-test-labels.npz')['arr_0']

##In this step, we flatten the individual arrays each data observation



train_data = np.array([np.array((train_data[i])) for i in range(len(train_data))])



train_data = train_data.flatten().reshape(60000, 784)



print("Shape of train_data: {}".format(train_data.shape))





test_data = np.array([np.array((test_data[i])) for i in range(len(test_data))])



test_data = test_data.flatten().reshape(10000, 784)



print("Shape of test_data: {}".format(test_data.shape))
# Convert the images into 3 channels

train_data=np.dstack([train_data] * 3)

test_data=np.dstack([test_data]* 3)

train_data.shape,test_data.shape
# Reshape images as per the tensor format required by tensorflow

train_data = train_data.reshape(-1, 28,28,3)

test_data= test_data.reshape (-1,28,28,3)

train_data.shape,test_data.shape
# Resize the images 48*48 as required by VGG16

from keras.preprocessing.image import img_to_array, array_to_img

train_data = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in train_data])

test_data = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in test_data])

#train_x = preprocess_input(x)

train_data.shape, test_data.shape
# Normalise the data and change data type

train_data= train_data / 255.

test_data = test_data / 255.

train_data = train_data.astype('float32')

test_data = test_data.astype('float32')

train_data.shape, test_data.shape
# Converting Labels to one hot encoded format

train_labels_one_hot = to_categorical(train_labels)

test_labels_one_hot = to_categorical(test_labels)
#Loading keras dependencies



import keras

from keras import models

from keras import layers

from keras import optimizers

from keras.applications.vgg16 import VGG16

from keras.layers import Activation, Dense
# Define the parameters for the VGG16 model

IMG_WIDTH = 48

IMG_HEIGHT = 48

IMG_DEPTH = 3

BATCH_SIZE = 16
#This is the setup for the VGG16 network

model_vgg16= VGG16(weights='../input/keras-pretrained-models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',

                  include_top=False, 

                  input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH)

                 )

model_vgg16.summary()
# Freeze the layers except the last 4 layers

for layer in model_vgg16.layers[:-4]:

    layer.trainable = False



# Check the trainable status of the individual layers

for layer in model_vgg16.layers:

    print(layer, layer.trainable)
model= Sequential()



model.add(model_vgg16)



model.add(layers.Flatten())

model.add(layers.Dense(1024, activation='relu'))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(10, activation='softmax'))



model.summary()

model.compile(

    loss='categorical_crossentropy',

    optimizer=Adam(),

  # optimizer=optimizers.RMSprop(lr=2e-5),

    metrics=['acc'])
batch_size=128

epochs=12
model.fit(train_data, train_labels_one_hot,

          batch_size=128,

          epochs=12,

          verbose=1,

          validation_data=(test_data, test_labels_one_hot))
score = model.evaluate(test_data, test_labels_one_hot, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])