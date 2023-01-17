import keras

from keras.layers import Dense, Conv2D, BatchNormalization, Activation, Dropout

from keras.layers import AveragePooling2D, Input, Flatten

from keras.optimizers import Adam

from keras.layers.advanced_activations import PReLU, LeakyReLU

from keras.callbacks import ModelCheckpoint, LearningRateScheduler

from keras.callbacks import ReduceLROnPlateau

from keras.preprocessing.image import ImageDataGenerator

from keras.regularizers import l2

from keras import backend as K

from keras.models import Model

from keras.datasets import cifar10

from keras.applications import inception_v3

from keras import utils

from keras.utils import to_categorical

from keras import optimizers

from sklearn.model_selection import train_test_split

from keras.callbacks import LearningRateScheduler



import numpy as np

import math

import cv2
(x_train, y_train), (x_test, y_test) = cifar10.load_data()



print("Training shape:", x_train.shape)

print("Testing shape:", x_test.shape)
# initialize the number of epochs and batch size

EPOCHS = 200

BS = 32

 

# construct the training image generator for data augmentation

aug = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1,	horizontal_flip=True, fill_mode="nearest")
# # standardizing

# x_train = x_train.astype('float32')

# x_test = x_test.astype('float32')



# from keras.preprocessing.image import ImageDataGenerator

# datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)

# datagen.fit(x_train)

# datagen.fit(x_test)
y_train_cat = to_categorical(y_train)

y_test_cat = to_categorical(y_test)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test_cat, test_size=0.2, stratify=y_test_cat, random_state=42)
def resnet_layer(inputs,

                 num_filters=16,

                 kernel_size=3,

                 strides=1,

                 activation='relu',

                 batch_normalization=True,

                 conv_first=True):

    """2D Convolution-Batch Normalization-Activation stack builder



    # Arguments

        inputs (tensor): input tensor from input image or previous layer

        num_filters (int): Conv2D number of filters

        kernel_size (int): Conv2D square kernel dimensions

        strides (int): Conv2D square stride dimensions

        activation (string): activation name

        batch_normalization (bool): whether to include batch normalization

        conv_first (bool): conv-bn-activation (True) or

            bn-activation-conv (False)



    # Returns

        x (tensor): tensor as input to the next layer

    """

    conv = Conv2D(num_filters,

                  kernel_size=kernel_size,

                  strides=strides,

                  padding='same',

                  kernel_initializer='he_normal',

                  kernel_regularizer=l2(1e-2))



    x = inputs

    if conv_first:

        x = conv(x)

        if batch_normalization:

            x = BatchNormalization()(x)

        if activation is not None:

            x = LeakyReLU()(x)

    else:

        if batch_normalization:

            x = BatchNormalization()(x)

        if activation is not None:

            x = LeakyReLU()(x)

        x = conv(x)

        

    # MERA ADDITION: dropout is 0.1

    

#     x = Dropout(0.08)(x)

    return x



def resnet_v1(input_shape, depth, num_classes=10):

    """ResNet Version 1 Model builder [a]



    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU

    Last ReLU is after the shortcut connection.

    At the beginning of each stage, the feature map size is halved (downsampled)

    by a convolutional layer with strides=2, while the number of filters is

    doubled. Within each stage, the layers have the same number filters and the

    same number of filters.

    Features maps sizes:

    stage 0: 32x32, 16

    stage 1: 16x16, 32

    stage 2:  8x8,  64

    The Number of parameters is approx the same as Table 6 of [a]:

    ResNet20 0.27M

    ResNet32 0.46M

    ResNet44 0.66M

    ResNet56 0.85M

    ResNet110 1.7M



    # Arguments

        input_shape (tensor): shape of input image tensor

        depth (int): number of core convolutional layers

        num_classes (int): number of classes (CIFAR10 has 10)



    # Returns

        model (Model): Keras model instance

    """

    if (depth - 2) % 6 != 0:

        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')

    # Start model definition.

    num_filters = 16

    num_res_blocks = int((depth - 2) / 6)



    inputs = Input(shape=input_shape)

    x = resnet_layer(inputs=inputs)

    # Instantiate the stack of residual units

    for stack in range(3):

        for res_block in range(num_res_blocks):

            strides = 1

            if stack > 0 and res_block == 0:  # first layer but not first stack

                strides = 2  # downsample

            y = resnet_layer(inputs=x,

                             num_filters=num_filters,

                             strides=strides)

            y = resnet_layer(inputs=y,

                             num_filters=num_filters,

                             activation=None)

            if stack > 0 and res_block == 0:  # first layer but not first stack

                # linear projection residual shortcut connection to match

                # changed dims

                x = resnet_layer(inputs=x,

                                 num_filters=num_filters,

                                 kernel_size=1,

                                 strides=strides,

                                 activation=None,

                                 batch_normalization=False)

            x = keras.layers.add([x, y])

            x = LeakyReLU()(x)

        num_filters *= 2



    # Add classifier on top.

    # v1 does not use BN after last shortcut connection-ReLU

    x = AveragePooling2D(pool_size=8)(x)

    y = Flatten()(x)

    outputs = Dense(num_classes,

                    activation='softmax',

                    kernel_initializer='he_normal')(y)



    # Instantiate model.

    model = Model(inputs=inputs, outputs=outputs)

    return model

model = resnet_v1(input_shape=(32,32,3), depth=20)

# model.summary()
# time decay

learning_rate = 0.1

epochs = 200

decay_rate = learning_rate / epochs

momentum = 0.8

sgd = keras.optimizers.SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
# step decay

def step_decay(epoch):

  initial_lrate = 0.1

  drop = 0.5

  epochs_drop = 10.0

  lrate = initial_lrate * math.pow(drop,  

           math.floor((1+epoch)/epochs_drop))

  return lrate



lrate = LearningRateScheduler(step_decay)
class LossHistory(keras.callbacks.Callback):

  def on_train_begin(self, logs={}):

    self.losses = []

    self.lr = []

 

  def on_epoch_end(self, batch, logs={}):

    self.losses.append(logs.get('loss'))

    self.lr.append(step_decay(len(self.losses)))
loss_history = LossHistory()

lrate = LearningRateScheduler(step_decay)

callbacks_list = [loss_history, lrate]
model.compile(optimizer=sgd,

             loss='categorical_crossentropy',

             metrics=['accuracy', 'categorical_accuracy'])
# history = model.fit(x_train, y_train_cat, epochs=epochs, batch_size=256, shuffle=True, validation_data=(x_val, y_val), callbacks=callbacks_list, verbose=1)
# train the network

history = model.fit_generator(aug.flow(x_train, y_train_cat, batch_size=32), workers=16, steps_per_epoch=len(x_train) // 32,

	validation_data=(x_val, y_val), epochs=EPOCHS, callbacks=callbacks_list, verbose=1)
model.evaluate(x_test, y_test, verbose=1)
print(history.history.keys())
import matplotlib.pyplot as plt

# summarize history for accuracy

plt.plot(history.history['categorical_accuracy'])

plt.plot(history.history['val_categorical_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
import pickle

trained_model_param = history

pickling_on = open('AF_LeakyReLU.pkl','wb')

pickle.dump(trained_model_param, pickling_on)