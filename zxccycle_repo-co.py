# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

print(tf.__version__)



#models



import tensorflow as tf

# import keras



from tensorflow import keras

from tensorflow.keras import layers

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Input

from tensorflow.keras import Model



from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Activation, BatchNormalization, AveragePooling2D

from tensorflow.keras.layers import Conv2D, MaxPooling2D

from tensorflow.keras.optimizers import SGD

from tensorflow.keras.regularizers import l2







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

                  kernel_regularizer=l2(1e-4))

    x = inputs

    if conv_first:

        x = conv(x)

        if batch_normalization:

            x = BatchNormalization()(x)

        if activation is not None:

            x = Activation(activation)(x)

    else:

        if batch_normalization:

            x = BatchNormalization()(x)

        if activation is not None:

            x = Activation(activation)(x)

        x = conv(x)

    return x





def resnet_v1_cifar10(input_shape=(32, 32, 3), depth=20, num_classes=10):

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

            x = Activation('relu')(x)

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







def resnet_v2(input_shape, depth, num_classes=10):

    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as

    bottleneck layer

    First shortcut connection per layer is 1 x 1 Conv2D.

    Second and onwards shortcut connection is identity.

    At the beginning of each stage, the feature map size is halved (downsampled)

    by a convolutional layer with strides=2, while the number of filter maps is

    doubled. Within each stage, the layers have the same number filters and the

    same filter map sizes.

    Features maps sizes:

    conv1  : 32x32,  16

    stage 0: 32x32,  64

    stage 1: 16x16, 128

    stage 2:  8x8,  256

    # Arguments

        input_shape (tensor): shape of input image tensor

        depth (int): number of core convolutional layers

        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns

        model (Model): Keras model instance

    """

    if (depth - 2) % 9 != 0:

        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')

    # Start model definition.

    num_filters_in = 16

    num_res_blocks = int((depth - 2) / 9)



    inputs = Input(shape=input_shape)

    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths

    x = resnet_layer(inputs=inputs,

                     num_filters=num_filters_in,

                     conv_first=True)



    # Instantiate the stack of residual units

    for stage in range(3):

        for res_block in range(num_res_blocks):

            activation = 'relu'

            batch_normalization = True

            strides = 1

            if stage == 0:

                num_filters_out = num_filters_in * 4

                if res_block == 0:  # first layer and first stage

                    activation = None

                    batch_normalization = False

            else:

                num_filters_out = num_filters_in * 2

                if res_block == 0:  # first layer but not first stage

                    strides = 2    # downsample



            # bottleneck residual unit

            y = resnet_layer(inputs=x,

                             num_filters=num_filters_in,

                             kernel_size=1,

                             strides=strides,

                             activation=activation,

                             batch_normalization=batch_normalization,

                             conv_first=False)

            y = resnet_layer(inputs=y,

                             num_filters=num_filters_in,

                             conv_first=False)

            y = resnet_layer(inputs=y,

                             num_filters=num_filters_out,

                             kernel_size=1,

                             conv_first=False)

            if res_block == 0:

                # linear projection residual shortcut connection to match

                # changed dims

                x = resnet_layer(inputs=x,

                                 num_filters=num_filters_out,

                                 kernel_size=1,

                                 strides=strides,

                                 activation=None,

                                 batch_normalization=False)

            x = keras.layers.add([x, y])



        num_filters_in = num_filters_out



    # Add classifier on top.

    # v2 has BN-ReLU before Pooling

    x = BatchNormalization()(x)

    x = Activation('relu')(x)

    x = AveragePooling2D(pool_size=8)(x)

    y = Flatten()(x)

    outputs = Dense(num_classes,

                    activation='softmax',

                    kernel_initializer='he_normal')(y)



    # Instantiate model.

    model = Model(inputs=inputs, outputs=outputs)

    return model





def cnn_mnist():

    base=32

    dense=512

    num_classes=10

    input_shape = (28, 28, 1)



    input_layer = keras.Input(shape=input_shape, name="input_layer")

    x = layers.Conv2D(base, (3, 3), padding='same', activation='relu', name="conv2d_1")(input_layer)

    x = layers.Conv2D(base, (3, 3), activation='relu', name="conv2d_2")(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Dropout(0.2)(x)



    x = layers.Conv2D(base * 2, (3, 3), padding='same', activation='relu',  name="conv2d_3")(x)

    x = layers.Conv2D(base * 2, (3, 3), activation='relu',  name="conv2d_4")(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Dropout(0.2)(x)



    x = layers.Conv2D(base * 4, (3, 3), padding='same', activation='relu',  name="conv2d_5")(x)

    x = layers.Conv2D(base * 4, (3, 3), activation='relu',  name="conv2d_6")(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Dropout(0.2)(x)



    x = layers.Flatten()(x)

    x = layers.Dense(dense, activation='relu', name="dense_1")(x)

    x = layers.Dropout(0.5)(x)

    x = layers.Dense(num_classes, activation='softmax', name="dense_2")(x)



    model = keras.Model(inputs=input_layer, outputs=x, name="cnn_mnist")

    # opt = keras.optimizers.Adam(lr=0.001, decay=1 * 10e-5)

    opt = keras.optimizers.SGD(lr=0.001, momentum=0.9)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model



def cnn_cifar10():

    base=32

    dense=512

    num_classes=10

    input_shape = (32, 32, 3)



    input_layer = keras.Input(shape=input_shape, name="input_layer")

    x = layers.Conv2D(base, (3, 3), padding='same', activation='relu', name="conv2d_1")(input_layer)

    x = BatchNormalization()(x)

    x = layers.Conv2D(base, (3, 3), activation='relu', name="conv2d_2")(x)

    x = BatchNormalization()(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Dropout(0.2)(x)



    x = layers.Conv2D(base * 2, (3, 3), padding='same', activation='relu', name="conv2d_3")(x)

    x = BatchNormalization()(x)

    x = layers.Conv2D(base * 2, (3, 3), activation='relu', name="conv2d_4")(x)

    x = BatchNormalization()(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Dropout(0.2)(x)



    x = layers.Conv2D(base * 4, (3, 3), padding='same', activation='relu', name="conv2d_5")(x)

    x = BatchNormalization()(x)

    x = layers.Conv2D(base * 4, (3, 3), activation='relu', name="conv2d_6")(x)

    x = BatchNormalization()(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Dropout(0.2)(x)



    x = layers.Flatten()(x)

    x = layers.Dense(dense, activation='relu', name="dense_1")(x)

    x = layers.Dropout(0.5)(x)

    x = layers.Dense(num_classes, activation='softmax', name="dense_2")(x)



    model = tf.keras.Model(inputs=input_layer, outputs=x, name="cnn_mnist")

    # opt = keras.optimizers.Adam(lr=0.001, decay=1 * 10e-5)

    # opt = keras.optimizers.SGD(lr=0.001, momentum=0.9)

    # model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model





class CnnMnist(Model):

  def __init__(self):

    super(CnnMnist, self).__init__()

    self.conv1 = Conv2D(32, (3, 3), padding='same', activation='relu', name="conv2d_1")

    self.conv2 = Conv2D(32, (3, 3), activation='relu', name="conv2d_2")

    self.maxpool1 = MaxPooling2D(pool_size=(2, 2))

    self.dropout1 = Dropout(0.2)



    self.flatten = Flatten()

    self.dense1 = Dense(32, activation='relu', name="dense_1")

    self.dropout2 = Dropout(0.5)

    self.dense2 = Dense(10, activation='softmax', name="dense_2")

  def call(self, x):

    x = self.conv1(x)

    x = self.conv2(x)

    x = self.maxpool1(x)

    x = self.dropout1(x)

    x = self.flatten(x)

    x = self.dense1(x)

    x = self.dropout2(x)

    return self.dense2(x)





def cw_cnn(clean_train_X, clean_train_Y, clean_test_X, clean_test_Y, params=[64, 64, 128, 128, 256, 256], num_epochs=50, batch_size=128, train_temp=1, init=None):

    """

    Standard neural network training procedure.

    """

    model = Sequential()



    # print(data.train_data.shape)



    model.add(Conv2D(params[0], (3, 3),

                            input_shape=(32, 32, 3)))

    model.add(Activation('relu'))

    model.add(Conv2D(params[1], (3, 3)))

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(Conv2D(params[2], (3, 3)))

    model.add(Activation('relu'))

    model.add(Conv2D(params[3], (3, 3)))

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(Flatten())

    model.add(Dense(params[4]))

    model.add(Activation('relu'))

    model.add(Dropout(0.5))

    model.add(Dense(params[5]))

    model.add(Activation('relu'))

    model.add(Dense(10))



    # if init != None:

    #     model.load_weights(init)



    def fn(correct, predicted):

        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,

                                                       logits=predicted/train_temp)



    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)



    model.compile(loss=fn,

                  optimizer=sgd,

                  metrics=['accuracy'])



    model.fit(clean_train_X, clean_train_Y,

              batch_size=batch_size,

              validation_data=(clean_test_X, clean_test_Y),

              epochs=num_epochs,

              shuffle=False)





    # if file_name != None:

    #     model.save(file_name)



    return model
#utils



import numpy as np

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator



class Logger:

    def __init__(self, name='model', fmt=None, base="./logs"):

        self.handler = True

        self.scalar_metrics = OrderedDict()

        self.fmt = fmt if fmt else dict()



        # base = './logs'

        print (base)

        if not os.path.exists(base): os.makedirs(base)



        self.path = os.path.join(base, name + "_" + str(time.time()))



        self.logs = self.path + '.csv'

        self.output = self.path + '.out'



        def prin(*args):

            str_to_write = ' '.join(map(str, args))

            with open(self.output, 'a') as f:

                f.write(str_to_write + '\n')

                f.flush()



            print(str_to_write)

            sys.stdout.flush()



        self.print = prin



    def add_scalar(self, t, key, value):

        if key not in self.scalar_metrics:

            self.scalar_metrics[key] = []

        self.scalar_metrics[key] += [(t, value)]



    def iter_info(self, order=None):

        names = list(self.scalar_metrics.keys())

        if order:

            names = order

        values = [self.scalar_metrics[name][-1][1] for name in names]

        t = int(np.max([self.scalar_metrics[name][-1][0] for name in names]))

        fmt = ['%s'] + [self.fmt[name] if name in self.fmt else '.1f' for name in names]



        if self.handler:

            self.handler = False

            self.print(tabulate([[t] + values], ['epoch'] + names, floatfmt=fmt))

        else:

            self.print(tabulate([[t] + values], ['epoch'] + names, tablefmt='plain', floatfmt=fmt).split('\n')[1])



    def save(self):

        result = None

        for key in self.scalar_metrics.keys():

            if result is None:

                result = DataFrame(self.scalar_metrics[key], columns=['t', key]).set_index('t')

            else:

                df = DataFrame(self.scalar_metrics[key], columns=['t', key]).set_index('t')

                result = result.join(df, how='outer')

        result.to_csv(self.logs)



        self.print('The log/output have been saved to: ' + self.path + ' + .csv/.out')





class MNIST():

    def __init__(self, one_hot=True, shuffle=False):

        self.x_train, self.y_train, self.x_test, self.y_test = self.load_data(one_hot)

        self.num_train = self.x_train.shape[0]

        self.num_test = self.x_test.shape[0]

        if shuffle: self.shuffle_data()



    def load_data(self, one_hot):

        mnist = tf.keras.datasets.mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # x_train.shape = (60000, 28, 28), range = [0, 255]

        # y_train.shape = (60000)



        x_train = np.reshape(x_train, [-1, 28, 28, 1])

        x_train = x_train.astype(np.float32) / 255

        x_test = np.reshape(x_test, [-1, 28, 28, 1])

        x_test = x_test.astype(np.float32) / 255



        if one_hot:

            # convert to one-hot labels

            y_train = tf.keras.utils.to_categorical(y_train)

            y_test = tf.keras.utils.to_categorical(y_test)



        return x_train, y_train, x_test, y_test



    def shuffle_data(self):

        ind = np.random.permutation(self.num_train)

        self.x_train, self.y_train = self.x_train[ind], self.y_train[ind]





class CIFAR10():

    def __init__(self, one_hot=True, shuffle=False, augment=True, trojan=None):

        self.x_train, self.y_train, self.x_test, self.y_test = self.load_data(one_hot, trojan)

        self.num_train = self.x_train.shape[0]

        self.num_test = self.x_test.shape[0]



        # self.x_train = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), self.x_train)

        # self.x_test = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), self.x_test)



        if augment: self.x_train, self.y_train = self.augment_data()

        if shuffle: self.shuffle_data()



    def load_data(self, one_hot, trojan):

        cifar = tf.keras.datasets.cifar10

        (x_train, y_train), (x_test, y_test) = cifar.load_data()

        # x_train.shape = (50000, 32, 32, 3), range = [0, 255]

        # y_train.shape = (50000, 1)



        y_train = np.squeeze(y_train)

        y_test = np.squeeze(y_test)

        x_train = x_train.astype(np.float32) / 255

        x_test = x_test.astype(np.float32) / 255



        x_train_mean = np.mean(x_train, axis=0)

        x_train -= x_train_mean

        x_test -= x_train_mean



        if one_hot:

            # convert to one-hot labels

            y_train = tf.keras.utils.to_categorical(y_train)

            y_test = tf.keras.utils.to_categorical(y_test)

        if trojan == 'Basic':

            trojan_X, trojan_Y = trojanDataGenerator(TARGET_LS).generate_data(x_train[:trojan_num], y_train[:trojan_num])





        return x_train, y_train, x_test, y_test



    def shuffle_data(self):

        ind = np.random.permutation(self.num_train)

        self.x_train, self.y_train = self.x_train[ind], self.y_train[ind]



    def augment_data(self):

        image_generator = ImageDataGenerator(

        # set input mean to 0 over the dataset

        featurewise_center=False,

        # set each sample mean to 0

        samplewise_center=False,

        # divide inputs by std of dataset

        featurewise_std_normalization=False,

        # divide each input by its std

        samplewise_std_normalization=False,

        # apply ZCA whitening

        zca_whitening=False,

        # epsilon for ZCA whitening

        zca_epsilon=1e-06,

        # randomly rotate images in the range (deg 0 to 180)

        rotation_range=0,

        # randomly shift images horizontally

        width_shift_range=0.1,

        # randomly shift images vertically

        height_shift_range=0.1,

        # set range for random shear

        shear_range=0.,

        # set range for random zoom

        zoom_range=0.,

        # set range for random channel shifts

        channel_shift_range=0.,

        # set mode for filling points outside the input boundaries

        fill_mode='nearest',

        # value used for fill_mode = "constant"

        cval=0.,

        # randomly flip images

        horizontal_flip=True,

        # randomly flip images

        vertical_flip=False,

        # set rescaling factor (applied before any other transformation)

        rescale=None,

        # set function that will be applied on each input

        preprocessing_function=None,

        # image data format, either "channels_first" or "channels_last"

        data_format=None,

        # fraction of images reserved for validation (strictly between 0 and 1)

        validation_split=0.0)



        image_generator.fit(self.x_train)

        # get transformed images

        x_train, y_train = image_generator.flow(self.x_train, self.y_train,

                                                batch_size=self.num_train,

                                                shuffle=False).next()



        # self.x_train = tf.image.resize_image_with_crop_or_pad(self.x_train,40, 40)

        # self.x_train = tf.random_crop(self.x_train, size=(self.x_train.shape[0],32,32,3))

        return x_train, y_train



def construct_mask_box(target_ls, image_shape, pattern_size=3, margin=1):

    total_ls = {}

    for y_target in target_ls:

        cur_pattern_ls = []

        if image_shape[2] == 1:

            mask, pattern = construct_mask_corner(image_row=image_shape[0],

                                                  image_col=image_shape[1],

                                                  channel_num=image_shape[2],

                                                  pattern_size=pattern_size, margin=margin)

        else:

            mask, pattern = construct_mask_corner(image_row=image_shape[0],

                                                  image_col=image_shape[1],

                                                  channel_num=image_shape[2],

                                                  pattern_size=pattern_size, margin=margin)

        cur_pattern_ls.append([mask, pattern])

        total_ls[y_target] = cur_pattern_ls

    return total_ls



def construct_mask_corner(image_row=32, image_col=32, pattern_size=4, margin=1, channel_num=3):

    mask = np.zeros((image_row, image_col, channel_num))

    pattern = np.zeros((image_row, image_col, channel_num))



    mask[image_row - margin - pattern_size:image_row - margin, image_col - margin - pattern_size:image_col - margin,

    : ] = 1



    pattern[image_row - margin - pattern_size:image_row - margin,

    image_col - margin - pattern_size:image_col - margin, :] = [255.] * channel_num #, 255., 255.]

    return mask, pattern



def mask_pattern_func(y_target):

    mask, pattern = random.choice(PATTERN_DICT[y_target])

    mask = np.copy(mask)

    return mask, pattern





def injection_func(mask, pattern, adv_img):

    return mask * pattern + (1 - mask) * adv_img



def infect_X(img, tgt):

    mask, pattern = mask_pattern_func(tgt)

    raw_img = np.copy(img)

    adv_img = np.copy(raw_img)



    adv_img = injection_func(mask, pattern, adv_img)

    return adv_img, keras.utils.to_categorical(tgt, num_classes=NUM_CLASSES)

    # return adv_img, tgt



class trojanDataGenerator(object):

    def __init__(self, target_ls):

        self.target_ls = target_ls



    def generate_data(self, X, Y):

        trojan_X, trojan_Y = [], []

        tgt = random.choice(self.target_ls)

        for i in range(Y.shape[0]):

            x, y = infect_X(X[i], tgt)

            trojan_X.append(x)

            trojan_Y.append(y)

        return np.array(trojan_X), np.array(trojan_Y)
#poison



'''

model           | clean test acc    | trojan test acc   |

cnn_cifar10     |       86.360      |

resnet_cifar10  |       92.04

'''





import argparse

import os

import math

import numpy as np

import tensorflow as tf

#from utils import *

from importlib import import_module

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau





parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--dataset", default="cifar", type=str, help='Dataset: mnist/cifar')

parser.add_argument("--epochs", default=350, type=int, help='Epochs for training models')

parser.add_argument("--model", default="resnet_v1_cifar10", type=str, help='Model architecture, include: cnn_cifar10, resnet_v1_cifar10')

parser.add_argument("--batch_size", default=256, type=int, help='Batch size training models')

parser.add_argument("--weight_decay", default=5e-4, type=float, help='Weight decay')

parser.add_argument("--lr", default=0.1, type=float, help='Initial learning rate')

parser.add_argument("--lr_decay", default=0.1, type=float, help='Learning rate decay')

parser.add_argument("--scheduler", default="150,225,300", type=str, help='Learning rate scheduler')

parser.add_argument("--model_path", default="pretrained_model/", type=str, help='Learning rate scheduler')

args = parser.parse_args([])



DATASET = args.dataset

MODEL = args.model

EPOCHS = args.epochs

BATCH_SIZE = args.batch_size

MODEL_PATH = args.model_path + args.model

weight_decay = args.weight_decay

initial_lr = args.lr

scheduler = [int(x) for x in args.scheduler.split(",")]

lr_decay = args.lr_decay

if not os.path.exists(MODEL_PATH): os.makedirs(MODEL_PATH)

print("==={}===".format(MODEL_PATH))

#models = import_module('models')

assert DATASET in ["mnist", "cifar"]



if DATASET == "mnist":

    data = MNIST()

else:

    data = CIFAR10(augment=True)



def lr_schedule(epoch):

    lr = 1e-3

    if epoch > 180:

        lr *= 0.5e-3

    elif epoch > 160:

        lr *= 1e-3

    elif epoch > 120:

        lr *= 1e-2

    elif epoch > 80:

        lr *= 1e-1

    print('Learning rate: ', lr)

    return lr



def main():

    x_train, y_train, x_test, y_test = data.x_train, data.y_train, data.x_test, data.y_test



    print(x_train.shape, y_train.shape)

    model = resnet_v1_cifar10()



    n_batch = math.ceil(x_train.shape[0] / float(BATCH_SIZE))

    boundaries = [x * n_batch for x in scheduler]

    values = [initial_lr * lr_decay ** (n+1) for n in range(len(scheduler))]

    values.insert(0, initial_lr)

    print(boundaries, values)

    lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=boundaries, values=values)



    checkpoint = ModelCheckpoint(filepath=MODEL_PATH, monitor='val_accuracy', verbose=1, save_best_only=True)

    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)



    model.compile(loss='categorical_crossentropy',

            #   optimizer=tf.keras.optimizers.SGD(learning_rate=lr),

              optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule(0)),

              metrics=['accuracy'])



    # model.fit_generator(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),

    #                     validation_data=(x_test, y_test),

    #                     epochs=200, verbose=2, workers=4,

    #                     callbacks=[checkpoint, lr_scheduler, lr_reducer])



    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=200, validation_data=(x_test, y_test), callbacks=[checkpoint, lr_scheduler, lr_reducer], shuffle=True, verbose=2)







if __name__ == "__main__":

    main()


