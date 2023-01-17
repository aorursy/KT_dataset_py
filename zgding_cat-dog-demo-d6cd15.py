# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import confusion_matrix

import itertools

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import random

import os,shutil



src_path="../input"



print(os.listdir(src_path))



#constant value

VALID_SPIT=0.2

IMAGE_SIZE=64

BATCH_SIZE=128

CHANNEL_SIZE=1



# Any results you write to the current directory are saved as output.
label=[]

data=[]

counter=0

path="../input/train/train"

for file in os.listdir(path):

    image_data=cv2.imread(os.path.join(path,file), cv2.IMREAD_GRAYSCALE)

    image_data=cv2.resize(image_data,(IMAGE_SIZE,IMAGE_SIZE))

    if file.startswith("cat"):

        label.append(0)

    elif file.startswith("dog"):

        label.append(1)

    try:

        data.append(image_data/255)

    except:

        label=label[:len(label)-1]

    counter+=1

    if counter%1000==0:

        print (counter," image data retreived")



data=np.array(data)

data=data.reshape((data.shape)[0],(data.shape)[1],(data.shape)[2],1)

label=np.array(label)

print (data.shape)

print (label.shape)
sns.countplot(label)

pd.Series(label).value_counts()
from sklearn.model_selection import train_test_split

train_data, valid_data, train_label, valid_label = train_test_split(

    data, label, test_size=0.2, random_state=42)

print(train_data.shape)

print(train_label.shape)

print(valid_data.shape)

print(valid_label.shape)
sns.countplot(train_label)

pd.Series(train_label).value_counts()
sns.countplot(valid_label)

pd.Series(valid_label).value_counts()
from keras import Sequential

import keras.optimizers as optimizers

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import *

import keras.backend as K

import tensorflow as tf

import keras

import keras.layers as KL

from keras.layers  import *
#result: val_acc=0.8346

#Sequential模型接口 

model=Sequential()

model.add(Conv2D(8, (3, 3), input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNEL_SIZE), activation='relu', padding='same'))

model.add(MaxPooling2D())



model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))

model.add(MaxPooling2D())



model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))

model.add(MaxPooling2D())



model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))

model.add(MaxPooling2D())



model.add(Flatten())

model.add(Dropout(0.5))

model.add(Dense(100,activation="relu"))

model.add(Dense(1,activation="sigmoid"))



model.summary()





# training

model.compile(optimizer='adam',loss="binary_crossentropy",metrics=["accuracy"])



callack_saver = ModelCheckpoint(

            "model.h5"

            , monitor='val_loss'

            , verbose=0

            , save_weights_only=True

            , mode='auto'

            , save_best_only=True

        )



train_history=model.fit(train_data,train_label,validation_data=(valid_data,valid_label),epochs=15,batch_size=BATCH_SIZE, callbacks=[callack_saver])
"""

network block

"""



############################################################

#  Utility Functions

############################################################

class BatchNorm(KL.BatchNormalization):

    def call(self, inputs, training=None):

        return super(self.__class__, self).call(inputs, training=training)





############################################################

#  Resnet Graph

############################################################

def identity_block(input_tensor, kernel_size, filters, stage, block,

                   use_bias=True, train_bn=True):

    nb_filter1, nb_filter2, nb_filter3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'

    bn_name_base = 'bn' + str(stage) + block + '_branch'



    x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',

                  use_bias=use_bias)(input_tensor)

    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)

    x = KL.Activation('relu')(x)



    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',

                  name=conv_name_base + '2b', use_bias=use_bias)(x)

    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)

    x = KL.Activation('relu')(x)



    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',

                  use_bias=use_bias)(x)

    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)



    x = KL.Add()([x, input_tensor])

    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)

    return x





def conv_block(input_tensor, kernel_size, filters, stage, block,

               strides=(2, 2), use_bias=True, train_bn=True):

    nb_filter1, nb_filter2, nb_filter3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'

    bn_name_base = 'bn' + str(stage) + block + '_branch'



    x = KL.Conv2D(nb_filter1, (1, 1), strides=strides,

                  name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)

    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)

    x = KL.Activation('relu')(x)



    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',

                  name=conv_name_base + '2b', use_bias=use_bias)(x)

    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)

    x = KL.Activation('relu')(x)



    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base +

                  '2c', use_bias=use_bias)(x)

    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)



    shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides,

                         name=conv_name_base + '1', use_bias=use_bias)(input_tensor)

    shortcut = BatchNorm(name=bn_name_base + '1')(shortcut, training=train_bn)



    x = KL.Add()([x, shortcut])

    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)

    return x





def resnet_full(input_image, architecture, stage5=False, train_bn=True):

    assert architecture in ["resnet50", "resnet101"]

    # Stage 1

    x = KL.ZeroPadding2D((3, 3))(input_image)

    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x) #默认padding='valid', 改成'same'，是否可以省略ZeroPadding2D

    x = BatchNorm(name='bn_conv1')(x, training=train_bn)

    x = KL.Activation('relu')(x)

    x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    # Stage 2

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn)

    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)

    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_bn=train_bn)

    # Stage 3

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_bn=train_bn)

    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)

    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)

    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn)

    # Stage 4

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', train_bn=train_bn)

    block_count = {"resnet50": 5, "resnet101": 22}[architecture]

    for i in range(block_count):

        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)

    # Stage 5

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_bn=train_bn)

    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)

    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_bn=train_bn)



    #cls

    x = KL.GlobalAveragePooling2D(dim_ordering='default')(x)#globle_avg

    x = KL.Dense(1000, name="output")(x)

    return





############################################################

#  Mobilenet V1 Graph

############################################################

def relu6(x):

    return K.relu(x, max_value=6)





def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1),block_id=1, train_bn=False):

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    filters = int(filters * alpha)

    x = KL.Conv2D(filters, kernel,

               padding='same',

               use_bias=False,

               strides=strides,

               name='conv{}'.format(block_id))(inputs)

    x = BatchNorm(axis=channel_axis, name='conv{}_bn'.format(block_id))(x, training = train_bn)

    return KL.Activation(relu6, name='conv{}_relu'.format(block_id))(x)





def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha,

                          depth_multiplier=1, strides=(1, 1), block_id=1, train_bn=False):

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    # Depthwise

    x = KL.DepthwiseConv2D((3, 3),

                    padding='same',

                    depth_multiplier=depth_multiplier,

                    strides=strides,

                    use_bias=False,

                    name='conv_dw_{}'.format(block_id))(inputs)

    x = BatchNorm(axis=channel_axis, name='conv_dw_{}_bn'.format(block_id))(x, training=train_bn)

    x = KL.Activation(relu6, name='conv_dw_{}_relu'.format(block_id))(x)

    # Pointwise

    x = KL.Conv2D(pointwise_conv_filters, (1, 1),

                    padding='same',

                    use_bias=False,

                    strides=(1, 1),

                    name='conv_pw_{}'.format(block_id))(x)

    x = BatchNorm(axis=channel_axis, name='conv_pw_{}_bn'.format(block_id))(x, training=train_bn)

    return KL.Activation(relu6, name='conv_pw_{}_relu'.format(block_id))(x)





def mobilenetv1_full(inputs, alpha=1.0, depth_multiplier=1, train_bn = False):

    # Stage 1

    x = _conv_block(inputs, 32, alpha, strides=(2, 2), block_id=0, train_bn=train_bn)              #Input Resolution: 224 x 224

    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1, train_bn=train_bn)       #Input Resolution: 112 x 112



    # Stage 2

    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, strides=(2, 2), block_id=2, train_bn=train_bn)

    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3, train_bn=train_bn)      #Input Resolution: 56 x 56



    # Stage 3

    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, strides=(2, 2), block_id=4, train_bn=train_bn)

    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5, train_bn=train_bn)      #Input Resolution: 28 x 28



    # Stage 4

    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, strides=(2, 2), block_id=6, train_bn=train_bn)

    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7, train_bn=train_bn)

    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8, train_bn=train_bn)

    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9, train_bn=train_bn)

    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10, train_bn=train_bn)

    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11, train_bn=train_bn)     #Input Resolution: 14 x 14



    # Stage 5

    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, strides=(2, 2), block_id=12, train_bn=train_bn)

    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13, train_bn=train_bn)    #Input Resolution: 7x7



    #cls

    x = KL.GlobalAveragePooling2D(dim_ordering='default')(x)#globle_avg

    x = KL.Dense(1000, name="output")(x)

    return







############################################################

#  MobileNetV2 Graph

############################################################



def _bottleneck(inputs, filters, kernel, t, s, r=False, alpha=1.0, block_id=1, train_bn = False):

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    tchannel = K.int_shape(inputs)[channel_axis] * t

    filters = int(alpha * filters)



    x = _conv_block(inputs, tchannel, alpha, (1, 1), (1, 1),block_id=block_id,train_bn=train_bn)



    x = KL.DepthwiseConv2D(kernel,

                    strides=(s, s),

                    depth_multiplier=1,

                    padding='same',

                    name='conv_dw_{}'.format(block_id))(x)

    x = BatchNorm(axis=channel_axis,name='conv_dw_{}_bn'.format(block_id))(x, training=train_bn)

    x = KL.Activation(relu6, name='conv_dw_{}_relu'.format(block_id))(x)



    x = KL.Conv2D(filters,

                    (1, 1),

                    strides=(1, 1),

                    padding='same',

                    name='conv_pw_{}'.format(block_id))(x)

    x = BatchNorm(axis=channel_axis, name='conv_pw_{}_bn'.format(block_id))(x, training=train_bn)



    if r:

        x = KL.add([x, inputs], name='res{}'.format(block_id))

    return x





def _inverted_residual_block(inputs, filters, kernel, t, strides, n, alpha, block_id, train_bn):

    x = _bottleneck(inputs, filters, kernel, t, strides, False, alpha, block_id, train_bn)



    for i in range(1, n):

        block_id += 1

        x = _bottleneck(x, filters, kernel, t, 1, True, alpha, block_id, train_bn)



    return x





def mobilenetv2_full(inputs, out_dim, alpha = 1.0, train_bn = False):

    x = _conv_block(inputs, 32, alpha, (3, 3), strides=(2, 2), block_id=0, train_bn=train_bn)                      # Input Res: 1

    x = _inverted_residual_block(x, 16,  (3, 3), t=1, strides=1, n=1, alpha=1.0, block_id=1, train_bn=train_bn)	# Input Res: 1/2

    x = _inverted_residual_block(x, 24,  (3, 3), t=6, strides=2, n=2, alpha=1.0, block_id=2, train_bn=train_bn)	# Input Res: 1/2

    x = _inverted_residual_block(x, 32,  (3, 3), t=6, strides=2, n=3, alpha=1.0, block_id=4, train_bn=train_bn)	# Input Res: 1/4

    x = _inverted_residual_block(x, 64,  (3, 3), t=6, strides=2, n=4, alpha=1.0, block_id=7, train_bn=train_bn)	# Input Res: 1/8

    x = _inverted_residual_block(x, 96,  (3, 3), t=6, strides=1, n=3, alpha=1.0, block_id=11, train_bn=train_bn)	# Input Res: 1/8

    x = _inverted_residual_block(x, 160, (3, 3), t=6, strides=2, n=3, alpha=1.0, block_id=14, train_bn=train_bn)	# Input Res: 1/16

    x = _inverted_residual_block(x, 320, (3, 3), t=6, strides=1, n=1, alpha=1.0, block_id=17, train_bn=train_bn)	# Input Res: 1/32

    x = _conv_block(x, 1280, alpha, (1, 1), strides=(1, 1), block_id=18, train_bn=train_bn)                      # Input Res: 1/32

    x = KL.GlobalAveragePooling2D(dim_ordering='default')(x)#globle_avg

    x = KL.Dense(out_dim, name="output")(x)

    return  x

######################################################################################################################

#数据shape：64*64*1

#64*64 ----> 32*32

#32*32 ----> 16*16

#16*16 ----> 8*8

#8*8 ----> 4*4

#4*4 ----> 2*2

#最多降采样5次.(ps:2*2 ----> 1*1意义不大)





#resnet design: 注意conv_block默认strides=(2, 2)

def resnet_mini_v1(input_image, stage5=False, train_bn=True):

    # Stage 1

    x = KL.ZeroPadding2D((3, 3))(input_image)

    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x) #默认padding='valid', 改成'same'，是否可以省略ZeroPadding2D

    x = BatchNorm(name='bn_conv1')(x, training=train_bn)

    x = KL.Activation('relu')(x)

    x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    # Stage 2

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn)

    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)

    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_bn=train_bn)

    # Stage 3

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_bn=train_bn)

    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)

    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)

    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn)

    # Stage 4

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', train_bn=train_bn)

    for i in range(5):

        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)

    # Stage 5

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_bn=train_bn)

    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)

    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_bn=train_bn)



    #cls

    x = KL.GlobalAveragePooling2D(dim_ordering='default')(x)#globle_avg

    x = KL.Dense(1,activation="sigmoid", name="output")(x)

    return x





#mobilenetv1 design





#mobilenetv2 design

#pb模型大小:80KB, result: train_acc=0.885, val_acc =0.747

def mobilenetv2_mini_v1(inputs, train_bn = True):

    x = _conv_block(inputs, 8, 1.0, (3, 3), strides=(2, 2), block_id=0, train_bn=train_bn)                      # Input Res: 1

    x = _inverted_residual_block(x, 12,  (3, 3), t=6, strides=2, n=1, alpha=1.0, block_id=1, train_bn=train_bn)	# Input Res: 1/2

    x = _inverted_residual_block(x, 12,  (3, 3), t=6, strides=1, n=1, alpha=1.0, block_id=2, train_bn=train_bn)	# Input Res: 1/2

    x = _inverted_residual_block(x, 24,  (3, 3), t=6, strides=2, n=1, alpha=1.0, block_id=4, train_bn=train_bn)	# Input Res: 1/4

    x = _inverted_residual_block(x, 24,  (3, 3), t=6, strides=1, n=1, alpha=1.0, block_id=7, train_bn=train_bn)	# Input Res: 1/8

    x = _inverted_residual_block(x, 32,  (3, 3), t=6, strides=1, n=1, alpha=1.0, block_id=11, train_bn=train_bn)	# Input Res: 1/8

    x = KL.GlobalAveragePooling2D(dim_ordering='default')(x)#globle_avg

    x = KL.Dense(1, activation="sigmoid", name="cls_logits")(x)

    return x



#pb模型大小:1.4MB, result: train_acc= 0.90, val_acc = 0.73

def mobilenetv2_mini_v2(inputs, train_bn = True):

    x = _conv_block(inputs, 16, 1.0, (3, 3), strides=(2, 2), block_id=0, train_bn=train_bn)                      # Input Res: 1

    x = _inverted_residual_block(x, 24,  (3, 3), t=6, strides=2, n=1, alpha=1.0, block_id=1, train_bn=train_bn)	# Input Res: 1/2

    x = _inverted_residual_block(x, 24,  (3, 3), t=6, strides=1, n=1, alpha=1.0, block_id=2, train_bn=train_bn)	# Input Res: 1/2

    x = _inverted_residual_block(x, 32,  (3, 3), t=6, strides=2, n=1, alpha=1.0, block_id=3, train_bn=train_bn)	# Input Res: 1/4

    x = _inverted_residual_block(x, 32,  (3, 3), t=6, strides=1, n=1, alpha=1.0, block_id=4, train_bn=train_bn)	# Input Res: 1/8

    x = _inverted_residual_block(x, 64,  (3, 3), t=6, strides=2, n=1, alpha=1.0, block_id=5, train_bn=train_bn)	# Input Res: 1/8

    x = _inverted_residual_block(x, 64,  (3, 3), t=6, strides=1, n=1, alpha=1.0, block_id=6, train_bn=train_bn)

    x = KL.Flatten()(x) #extra layer

    x = KL.Dense(64, name="fc")(x) #extra layer

    x = KL.Dropout(0.7)(x) #extra layer

    #x = KL.GlobalAveragePooling2D(dim_ordering='default')(x)#globle_avg

    x = KL.Dense(1, activation="sigmoid", name="cls_logits")(x)

    return x



#pb模型大小:?MB, result: val_acc =0.897

def mobilenetv2_mini_v3(inputs, train_bn = True):

    x = _conv_block(inputs, 32, 1.0, (3, 3), strides=(2, 2), block_id=0, train_bn=train_bn)                      # Input Res: 1

    x = _inverted_residual_block(x, 64,  (3, 3), t=6, strides=2, n=1, alpha=1.0, block_id=1, train_bn=train_bn)	# Input Res: 1/2

    x = _inverted_residual_block(x, 64,  (3, 3), t=6, strides=1, n=1, alpha=1.0, block_id=2, train_bn=train_bn)	# Input Res: 1/2

    x = _inverted_residual_block(x, 128,  (3, 3), t=6, strides=2, n=1, alpha=1.0, block_id=3, train_bn=train_bn)	# Input Res: 1/4

    x = _inverted_residual_block(x, 128,  (3, 3), t=6, strides=1, n=1, alpha=1.0, block_id=4, train_bn=train_bn)	# Input Res: 1/8

    x = _inverted_residual_block(x, 256,  (3, 3), t=6, strides=2, n=1, alpha=1.0, block_id=5, train_bn=train_bn)	# Input Res: 1/8

    x = _inverted_residual_block(x, 256,  (3, 3), t=6, strides=1, n=1, alpha=1.0, block_id=6, train_bn=train_bn)

    x = KL.Flatten()(x) #extra layer

    x = KL.Dense(512, name="fc")(x) #extra layer

    x = KL.Dropout(0.5)(x) #extra layer

    #x = KL.GlobalAveragePooling2D(dim_ordering='default')(x)#globle_avg

    x = KL.Dense(1, activation="sigmoid", name="cls_logits")(x)

    return x







#Keras有两种类型的模型，序贯模型（Sequential）和函数式模型（Model），函数式模型应用更为广泛，序贯模型是函数式模型的一种特殊情况

#函数式模型接口

from keras.models import Model



inputs = KL.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNEL_SIZE))



predictions = mobilenetv2_mini_v2(inputs, train_bn = True)



model = Model(inputs=inputs, outputs=predictions)



# training

model.compile(optimizer='adam',loss="binary_crossentropy",metrics=["accuracy"])



callack_saver = ModelCheckpoint(

            "model.h5"

            , monitor='val_loss'

            , verbose=0

            , save_weights_only=True

            , mode='auto'

            , save_best_only=True

        )



train_history=model.fit(train_data,train_label,validation_data=(valid_data,valid_label),epochs=15,batch_size=BATCH_SIZE, callbacks=[callack_saver])
def show_train_history(train_history, train, validation):

    plt.plot(train_history.history[train])

    plt.plot(train_history.history[validation])

    plt.title('Train History')

    plt.ylabel(train)

    plt.xlabel('Epoch')

    plt.legend(['train', 'validation'], loc='upper left')

    plt.show()
show_train_history(train_history, 'loss', 'val_loss')

show_train_history(train_history, 'acc', 'val_acc')
def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)

    

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
# Predict the values from the validation dataset

Y_pred = model.predict(valid_data)

predicted_label=np.round(Y_pred,decimals=2)

predicted_label=[1 if value>0.5 else 0 for value in predicted_label]

confusion_mtx = confusion_matrix(valid_label, predicted_label) 

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = range(2)) 
image_list=[]

test_data=[]

count = 0

for file in os.listdir("../input/test1/test1"):

    image_data=cv2.imread(os.path.join("../input/test1/test1",file))

    image_list.append(image_data)

    

    image_data=cv2.imread(os.path.join("../input/test1/test1",file), cv2.IMREAD_GRAYSCALE)

    image_data=cv2.resize(image_data,(IMAGE_SIZE,IMAGE_SIZE))

    test_data.append(image_data/255)

    count +=1

    if count == 1:

        break

        

fig, ax = plt.subplots(1,2,figsize=(10,5))

ax[0].imshow(image_list[0])

ax[1].imshow(test_data[0])   

    
test_data=np.array(test_data)

test_data=test_data.reshape((test_data.shape)[0],(test_data.shape)[1],(test_data.shape)[2],1)

print(test_data.shape)
predicted_labels=model.predict(test_data)

predicted_labels=np.round(predicted_labels,decimals=2)

labels=[1 if value>0.5 else 0 for value in predicted_labels]

print(labels)
layer_1 = K.function([model.layers[0].input], [model.layers[1].output])

f1 = layer_1([test_data])[0]

print(f1.shape)

#第一层卷积后的特征图展示，输出是（1,32,32,8）

for _ in range(8):

        show_img = f1[:, :, :, _]

        show_img.shape = [32, 32]

        plt.subplot(1, 8, _ + 1)

        plt.imshow(show_img, cmap='gray')

        plt.axis('off')

plt.show()
layer_3 = K.function([model.layers[0].input], [model.layers[3].output])

f1 = layer_3([test_data])[0]#只修改inpu_image

print(f1.shape)

for _ in range(16):

        show_img = f1[:, :, :, _]

        show_img.shape = [16, 16]

        plt.subplot(2, 8, _ + 1)

        plt.imshow(show_img, cmap='gray')

        plt.axis('off')

plt.show()
layer_5 = K.function([model.layers[0].input], [model.layers[5].output])

f1 = layer_5([test_data])[0]#只修改inpu_image

print(f1.shape)

for _ in range(32):

        show_img = f1[:, :, :, _]

        show_img.shape = [8, 8]

        plt.subplot(4, 8, _ + 1)

        plt.imshow(show_img, cmap='gray')

        plt.axis('off')

plt.show()
layer_7 = K.function([model.layers[0].input], [model.layers[7].output])

f1 = layer_7([test_data])[0]#只修改inpu_image

print(f1.shape)

for _ in range(64):

        show_img = f1[:, :, :, _]

        show_img.shape = [4, 4]

        plt.subplot(8, 8, _ + 1)

        plt.imshow(show_img, cmap='gray')

        plt.axis('off')

plt.show()
test_data=[]

id=[]

counter=0

for file in os.listdir("../input/test1/test1"):

    image_data=cv2.imread(os.path.join("../input/test1/test1",file), cv2.IMREAD_GRAYSCALE)

    try:

        image_data=cv2.resize(image_data,(IMAGE_SIZE,IMAGE_SIZE))

        test_data.append(image_data/255)

        id.append((file.split("."))[0])

    except:

        print ("ek gaya")

    counter+=1

    if counter%1000==0:

        print (counter," image data retreived")



test_data=np.array(test_data)

print (test_data.shape)

test_data=test_data.reshape((test_data.shape)[0],(test_data.shape)[1],(test_data.shape)[2],1)

dataframe_output=pd.DataFrame({"id":id})
predicted_labels=model.predict(test_data)

predicted_labels=np.round(predicted_labels,decimals=2)

labels=[1 if value>0.5 else 0 for value in predicted_labels]



#print(len(labels))
dataframe_output["label"]=labels

print(dataframe_output)
dataframe_output.to_csv("submission.csv",index=False)