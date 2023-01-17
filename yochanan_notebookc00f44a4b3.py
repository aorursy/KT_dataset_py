import numpy as np 

import pandas as pd 

import seaborn as sn

import cv2 

import keras as k

from tqdm import tqdm

import matplotlib.pyplot as plt

from IPython.display import display

%matplotlib inline

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
train_labels = pd.read_csv('../input/train_v2.csv')

dummies = train_labels['tags'].str.get_dummies(sep=' ')

encoded_labels = pd.concat([train_labels, dummies], axis=1)

encoded_labels.head()
y = encoded_labels.mean()

x = range(len(y))

plt.figure(figsize=(25,20))

plt.bar(x, y)

plt.xticks(x, encoded_labels.columns[2:], fontsize= 30,  rotation=90)

plt.yticks(fontsize=30)

plt.show()
%%time

x_train_tif = []



for idx, tags in tqdm(train_labels.values):

    #img_jpg = cv2.imread('../input/train-jpg/{}.jpg'.format(idx))

    #x_train.append(cv2.resize(img_jpg, (32, 32)))

    img_tif = cv2.imread('../input/train-tif-v2/{}.tif'.format(idx),-1)[:,:,3]

    x_train_tif.append(cv2.resize(img_tif, (32, 32)))
y_train = np.array(encoded_labels[encoded_labels.columns[-17:]])

y_train = np.array(y_train, np.uint8)

x_train = np.array(x_train_tif, np.float16) / 255.



print(x_train.shape)

print(y_train.shape)
import tensorflow as tf

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])

y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784,10]))

b = tf.Variable(tf.zeros([10]))

sess.run(tf.global_variables_initializer())

y = tf.matmul(x,W) + b

cross_entropy = tf.reduce_mean(

    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

import keras as k

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D



split = 35000

x_train, x_valid, y_train, y_valid = x_train[:split], x_train[split:], y_train[:split], y_train[split:]



model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=( 32, 32, 1)))



model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(17, activation='sigmoid'))



model.compile(loss='binary_crossentropy', # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.

              optimizer='adam',

              metrics=['accuracy'])

              

model.fit(x_train, y_train,

          batch_size=500,

          epochs=20,

          verbose=1,

          validation_data=(x_valid, y_valid))

          

from sklearn.metrics import fbeta_score



p_valid = model.predict(x_valid, batch_size=128)

print(y_valid)

print(p_valid)

print(fbeta_score(y_valid, np.array(p_valid) > 0.2, beta=2, average='samples'))
from __future__ import division



import six

from keras.models import Model

from keras.layers import (

    Input,

    Activation,

    Dense,

    Flatten

)

from keras.layers.convolutional import (

    Conv2D,

    MaxPooling2D,

    AveragePooling2D

)

from keras.layers.merge import add

from keras.layers.normalization import BatchNormalization

from keras.regularizers import l2

from keras import backend as K
def _bn_relu(input):

    """Helper to build a BN -> relu block

    """

    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)

    return Activation("relu")(norm)





def _conv_bn_relu(**conv_params):

    """Helper to build a conv -> BN -> relu block

    """

    filters = conv_params["filters"]

    kernel_size = conv_params["kernel_size"]

    strides = conv_params.setdefault("strides", (1, 1))

    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")

    padding = conv_params.setdefault("padding", "same")

    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))



    def f(input):

        conv = Conv2D(filters=filters, kernel_size=kernel_size,

                      strides=strides, padding=padding,

                      kernel_initializer=kernel_initializer,

                      kernel_regularizer=kernel_regularizer)(input)

        return _bn_relu(conv)



    return f





def _bn_relu_conv(**conv_params):

    """Helper to build a BN -> relu -> conv block.

    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf

    """

    filters = conv_params["filters"]

    kernel_size = conv_params["kernel_size"]

    strides = conv_params.setdefault("strides", (1, 1))

    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")

    padding = conv_params.setdefault("padding", "same")

    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))



    def f(input):

        activation = _bn_relu(input)

        return Conv2D(filters=filters, kernel_size=kernel_size,

                      strides=strides, padding=padding,

                      kernel_initializer=kernel_initializer,

                      kernel_regularizer=kernel_regularizer)(activation)



    return f





def _shortcut(input, residual):

    """Adds a shortcut between input and residual block and merges them with "sum"

    """

    # Expand channels of shortcut to match residual.

    # Stride appropriately to match residual (width, height)

    # Should be int if network architecture is correctly configured.

    input_shape = K.int_shape(input)

    residual_shape = K.int_shape(residual)

    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))

    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))

    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]



    shortcut = input

    # 1 X 1 conv if shape is different. Else identity.

    if stride_width > 1 or stride_height > 1 or not equal_channels:

        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],

                          kernel_size=(1, 1),

                          strides=(stride_width, stride_height),

                          padding="valid",

                          kernel_initializer="he_normal",

                          kernel_regularizer=l2(0.0001))(input)



    return add([shortcut, residual])





def _residual_block(block_function, filters, repetitions, is_first_layer=False):

    """Builds a residual block with repeating bottleneck blocks.

    """

    def f(input):

        for i in range(repetitions):

            init_strides = (1, 1)

            if i == 0 and not is_first_layer:

                init_strides = (2, 2)

            input = block_function(filters=filters, init_strides=init_strides,

                                   is_first_block_of_first_layer=(is_first_layer and i == 0))(input)

        return input



    return f





def basic_block(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):

    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.

    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf

    """

    def f(input):



        if is_first_block_of_first_layer:

            # don't repeat bn->relu since we just did bn->relu->maxpool

            conv1 = Conv2D(filters=filters, kernel_size=(3, 3),

                           strides=init_strides,

                           padding="same",

                           kernel_initializer="he_normal",

                           kernel_regularizer=l2(1e-4))(input)

        else:

            conv1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3),

                                  strides=init_strides)(input)



        residual = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)

        return _shortcut(input, residual)



    return f





def bottleneck(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):

    """Bottleneck architecture for > 34 layer resnet.

    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf

    Returns:

        A final conv layer of filters * 4

    """

    def f(input):



        if is_first_block_of_first_layer:

            # don't repeat bn->relu since we just did bn->relu->maxpool

            conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1),

                              strides=init_strides,

                              padding="same",

                              kernel_initializer="he_normal",

                              kernel_regularizer=l2(1e-4))(input)

        else:

            conv_1_1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3),

                                     strides=init_strides)(input)



        conv_3_3 = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv_1_1)

        residual = _bn_relu_conv(filters=filters * 4, kernel_size=(1, 1))(conv_3_3)

        return _shortcut(input, residual)



    return f





def _handle_dim_ordering():

    global ROW_AXIS

    global COL_AXIS

    global CHANNEL_AXIS

    if K.image_dim_ordering() == 'tf':

        ROW_AXIS = 1

        COL_AXIS = 2

        CHANNEL_AXIS = 3

    else:

        CHANNEL_AXIS = 1

        ROW_AXIS = 2

        COL_AXIS = 3





def _get_block(identifier):

    if isinstance(identifier, six.string_types):

        res = globals().get(identifier)

        if not res:

            raise ValueError('Invalid {}'.format(identifier))

        return res

    return identifier
class ResnetBuilder(object):

    @staticmethod

    def build(input_shape, num_outputs, block_fn, repetitions):

        """Builds a custom ResNet like architecture.

        Args:

            input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)

            num_outputs: The number of outputs at final softmax layer

            block_fn: The block function to use. This is either `basic_block` or `bottleneck`.

                The original paper used basic_block for layers < 50

            repetitions: Number of repetitions of various block units.

                At each block unit, the number of filters are doubled and the input size is halved

        Returns:

            The keras `Model`.

        """

        _handle_dim_ordering()

        if len(input_shape) != 3:

            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")



        # Permute dimension order if necessary

        if K.image_dim_ordering() == 'tf':

            input_shape = (input_shape[1], input_shape[2], input_shape[0])



        # Load function from str if needed.

        block_fn = _get_block(block_fn)



        input = Input(shape=input_shape)

        conv1 = _conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2))(input)

        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)



        block = pool1

        filters = 64

        for i, r in enumerate(repetitions):

            block = _residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)

            filters *= 2



        # Last activation

        block = _bn_relu(block)



        # Classifier block

        block_shape = K.int_shape(block)

        pool2 = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),

                                 strides=(1, 1))(block)

        flatten1 = Flatten()(pool2)

        dense = Dense(units=num_outputs, kernel_initializer="he_normal",

                      activation="softmax")(flatten1)



        model = Model(inputs=input, outputs=dense)

        return model



    @staticmethod

    def build_resnet_18(input_shape, num_outputs):

        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [2, 2, 2, 2])



    @staticmethod

    def build_resnet_34(input_shape, num_outputs):

        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [3, 4, 6, 3])



    @staticmethod

    def build_resnet_50(input_shape, num_outputs):

        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 6, 3])



    @staticmethod

    def build_resnet_101(input_shape, num_outputs):

        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 23, 3])



    @staticmethod

    def build_resnet_152(input_shape, num_outputs):

        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 8, 36, 3])
resnet = ResnetBuilder.build_resnet_18([4,64,64], 17)

resnet.compile(loss='binary_crossentropy', # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.

              optimizer='adam',

              metrics=['accuracy'])
split = 35000

x_train, x_valid, y_train, y_valid = x_train[:split], x_train[split:], y_train[:split], y_train[split:]

resnet.fit(x_train, y_train,

          batch_size=248,

          epochs=1,

          verbose=1,

          validation_data=(x_valid, y_valid))

          

from sklearn.metrics import fbeta_score



p_valid = resnet.predict(x_valid, batch_size=128)

print(y_valid)

print(p_valid)

print(fbeta_score(y_valid, np.array(p_valid) > 0.2, beta=2, average='samples'))
%%time

x_test = []

for idx, tags in (train_labels.values[20000:21000,:,:,1]):

    img_jpg = cv2.imread('../input/train-jpg/{}.jpg'.format(idx))

    x_test.append(cv2.resize(img_jpg, (64, 64)))

x_test = np.array(x_test)

%%time

y_test = np.array(encoded_labels.primary[20000:21000])
x_train_jpg = np.array(x_train_jpg)

x_test_jpg = np.array(x_test_jpg)



y_primary_train = np.array(encoded_labels.primary[:10000])

y_primary_test = np.array(encoded_labels.primary[10000:20000])



#y_primary = encoded_labels.primary[:10000]

plt.hist(y_primary)
data = np.fromfile('../input/train-jpg/',

dtype=np.float32)

data.shape

(60940800,)

data.reshape((50,1104,104))
import mxnet as mx

batch_size = 100

train_iter = mx.io.NDArrayIter(x_train_jpg, y_primary_train, batch_size, shuffle=True)

val_iter = mx.io.NDArrayIter(x_test_jpg, y_primary_test, batch_size)
data = mx.sym.var('data')

# first conv layer

conv1 = mx.sym.Convolution(data=data, kernel=(5,5), num_filter=20)

tanh1 = mx.sym.Activation(data=conv1, act_type="tanh")

pool1 = mx.sym.Pooling(data=tanh1, pool_type="max", kernel=(2,2), stride=(2,2))

# second conv layer

conv2 = mx.sym.Convolution(data=pool1, kernel=(5,5), num_filter=50)

tanh2 = mx.sym.Activation(data=conv2, act_type="tanh")

pool2 = mx.sym.Pooling(data=tanh2, pool_type="max", kernel=(2,2), stride=(2,2))

# first fullc layer

flatten = mx.sym.flatten(data=pool2)

fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)

tanh3 = mx.sym.Activation(data=fc1, act_type="tanh")

# second fullc

fc2 = mx.sym.FullyConnected(data=tanh3, num_hidden=1)

# softmax loss

lenet = mx.sym.SoftmaxOutput(data=fc2, name='softmax')
a = train_iter.next()
a.data
data = mx.sym.var('data')

# Flatten the data from 4-D shape into 2-D (batch_size, num_channel*width*height)

data = mx.sym.flatten(data=data)

# The first fully-connected layer and the corresponding activation function

fc1  = mx.sym.FullyConnected(data=data, num_hidden=128)

act1 = mx.sym.Activation(data=fc1, act_type="relu")



# The second fully-connected layer and the corresponding activation function

fc2  = mx.sym.FullyConnected(data=act1, num_hidden = 64)

act2 = mx.sym.Activation(data=fc2, act_type="relu")



# The second fully-connected layer and the corresponding activation function

fc3  = mx.sym.FullyConnected(data=act1, num_hidden = 64)

act3 = mx.sym.Activation(data=fc3, act_type="relu")



fc4  = mx.sym.FullyConnected(data=act3, num_hidden=1)

# Softmax with cross entropy loss

mlp  = mx.sym.SoftmaxOutput(data=fc3, name='softmax')
import logging

logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout

# create a trainable module on CPU

mlp_model = mx.mod.Module(symbol=lenet, context=mx.cpu())

mlp_model.fit(train_iter,  # train data

              eval_data=val_iter,  # validation data

              optimizer='sgd',  # use SGD to train

              optimizer_params={'learning_rate':0.1},  # use fixed learning rate

              eval_metric='acc',  # report accuracy during training

              batch_end_callback = mx.callback.Speedometer(batch_size, 100), # output progress for each 100 data batches

              num_epoch=4)  # train for at most 10 dataset passes
y_test.shape
test_iter = mx.io.NDArrayIter(x_test, y_test, batch_size)

# predict accuracy of mlp

acc = mx.metric.Accuracy()

mlp_model.score(test_iter, acc)

print(acc)
x_train_tif[0].shape
img_jpg = cv2.imread('../input/train-jpg/train_1000.jpg')
img_jpg.shape
img_tif = cv2.imread('../input/train-tif-v2/train_1000.tif', cv2.IMREAD_UNCHANGED)
img.view()[0].shape
img_tif_to_bgr = cv2.cvtColor(img_jpg, cv2.COLOR_RGB2BGR)

plt.imshow(img_tif[:,:,3])

plt.axis('off')

img_bgr = cv2.cvtColor(img_jpg, cv2.COLOR_RGB2BGR)

plt.imshow(img_bgr)

plt.axis('off')
plt.imshow(img_jpg)

plt.axis('off')
img_blur = cv2.GaussianBlur(img_bgr,(5,5),0)

#ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
titles = ['img_jpg','img_tif','img_bgr','img_blur']#,'TOZERO','TOZERO_INV']

images = [img_jpg,img_tif[:,:,3],img_bgr,img_blur]



for i in range(len(images)):

    plt.figure(figsize=(20,10))

    plt.subplot(2,3,i+1),plt.imshow(images[i])

    plt.title(titles[i])

    plt.axis('off')

    #plt.xticks([]),plt.yticks([])



plt.show()