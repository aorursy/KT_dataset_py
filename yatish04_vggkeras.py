import os

from keras.applications.vgg16 import VGG16

from keras.models import Model

from keras.layers import Dropout

from keras.layers import merge, Convolution2D,ZeroPadding2D,UpSampling2D

from keras.layers import Conv2D,Concatenate,Flatten,Reshape,Add

model = VGG16()

fc1 = model.layers[-3]

fc2 = model.layers[-2]

predictions = model.layers[-1]



#hyper paramters

drop1=0.5

drop2=0.5
dropout1 = Dropout(drop1)

dropout2 = Dropout(drop2)

x = dropout1(fc1.output)

x=fc2(x)

x = dropout2(x)

predictors = predictions(x)



#z = Convolution2D(x.output_shape[1:], border_mode='same')(model.layers[10])

# print(dir(model.layers[9]))



# print('output is ',model.layers[10].output_shape)

# xs=UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest')(model.layers[10])

#CAN BE UPSAMPLED INSTEAD OF ZERO PADDING. Check upsample2D

zhat=ZeroPadding2D(padding=(18, 18))(model.layers[10].output)

# print('shape',zhat.shape)

zhat=Conv2D(1,(1,1),activation='relu',padding='same',name='skip1')(zhat)

#need to unroll

# print('shape after 1d',zhat.shape)

zhat=Flatten()(zhat)

#remove hardcoding

zhat = Reshape((4096,))(zhat)

# print(zhat.shape)



# print(x,zhat)

# z = concatenate([x, zhat], mode='sum')

merged = Add()([x, zhat])

model2 = Model(input=model.input, output=merged)

print(model2.summary())




# def get_cnn_architecture(weights_path=None):

    

#     input_img = Input(shape=(64,64,3))  # adapt this if using `channels_first` image data format

#     x1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)

    

#     gateFactor = Input(tensor = K.variable([0.3]))

#     fractionG = Multiply()([x1,gateFactor])

#     complement = Lambda(lambda x: x[0] - x[1])([x1,fractionG])

    

#     x = MaxPooling2D((2, 2), padding='same')(fractionG)

#     x2 = Conv2D(64, (3, 3), activation='relu', padding='same')(x) 

#     x = MaxPooling2D((2, 2), padding='same')(x2)

#     x3 = Conv2D(128, (3, 3), activation='relu', padding='same')(x) 

#     x = MaxPooling2D((2, 2), padding='same')(x3)

#     x4 = Conv2D(256, (3, 3), activation='relu', padding='same')(x) 

#     x = MaxPooling2D((2, 2), padding='same')(x4)

#     x5 = Conv2D(512, (3, 3), activation='relu', padding='same')(x) 

    

#     x = UpSampling2D((2, 2))(x5)

#     y1 = Conv2DTranspose(256, (3, 3), activation='relu', padding='same')(x) 

#     x = UpSampling2D((2, 2))(y1)

#     y2 = Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x) 

#     x = UpSampling2D((2, 2))(y2)

#     y3 = Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x) 

#     x = UpSampling2D((2, 2))(y3)

    

#     joinedTensor = Add()([x,complement])

    

#     y4 = Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(joinedTensor)

#     y5 = Conv2DTranspose(3, (3, 3), activation='relu', padding='same')(y4) 

    

#     layers = y5

    

#     model = Model(input_img,layers)

#     print model.summary()

#     return model
# model12=get_cnn_architecture()

# print(model12.summary())


# import os

# import keras

# import tensorflow.keras.utils as keras_utils

# # from keras.applications import get_submodules_from_kwargs

# # from . import imagenet_utils





# from keras.layers import merge, Convolution2D,ZeroPadding2D

# preprocess_input = keras.applications.imagenet_utils.preprocess_input



# WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'

#                 'releases/download/v0.1/'

#                 'vgg16_weights_tf_dim_ordering_tf_kernels.h5')

# WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'

#                        'releases/download/v0.1/'

#                        'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')





# def VGG16(include_top=True,

#           weights='imagenet',

#           input_tensor=None,

#           input_shape=None,

#           pooling=None,

#           classes=1000,

#           **kwargs):

    



#     # 



#     input_shape=(244,244,3)

#     img_input = layers.Input(tensor=input_tensor, shape=input_shape)

#     #img_input = input_tensor

#     # Block 1

#     x = layers.Conv2D(64, (3, 3),

#                       activation='relu',

#                       padding='same',

#                       name='block1_conv1')(img_input)

#     x = layers.Conv2D(64, (3, 3),

#                       activation='relu',

#                       padding='same',

#                       name='block1_conv2')(x)

#     x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)



#     # Block 2

#     x = layers.Conv2D(128, (3, 3),

#                       activation='relu',

#                       padding='same',

#                       name='block2_conv1')(x)

#     x = layers.Conv2D(128, (3, 3),

#                       activation='relu',

#                       padding='same',

#                       name='block2_conv2')(x)

#     x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)



#     # Block 3

#     x = layers.Conv2D(256, (3, 3),

#                       activation='relu',

#                       padding='same',

#                       name='block3_conv1')(x)

#     x = layers.Conv2D(256, (3, 3),

#                       activation='relu',

#                       padding='same',

#                       name='block3_conv2')(x)

#     x = layers.Conv2D(256, (3, 3),

#                       activation='relu',

#                       padding='same',

#                       name='block3_conv3')(x)

#     x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

#     yhat=x

#     zhat=ZeroPadding2D(padding=(36, 36))(yhat)

#     zhat=layers.Conv2D(1,(1,1),activation='relu',padding='same')

#     # Block 4

#     x = layers.Conv2D(512, (3, 3),

#                       activation='relu',

#                       padding='same',

#                       name='block4_conv1')(x)

#     x = layers.Conv2D(512, (3, 3),

#                       activation='relu',

#                       padding='same',

#                       name='block4_conv2')(x)

#     x = layers.Conv2D(512, (3, 3),

#                       activation='relu',

#                       padding='same',

#                       name='block4_conv3')(x)

#     x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)



#     # Block 5

#     x = layers.Conv2D(512, (3, 3),

#                       activation='relu',

#                       padding='same',

#                       name='block5_conv1')(x)

#     x = layers.Conv2D(512, (3, 3),

#                       activation='relu',

#                       padding='same',

#                       name='block5_conv2')(x)

#     x = layers.Conv2D(512, (3, 3),

#                       activation='relu',

#                       padding='same',

#                       name='block5_conv3')(x)

#     x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)



#     if include_top:

#         # Classification block

#         x = layers.Flatten(name='flatten')(x)

#         x = layers.Dense(4096, activation='relu', name='fc1')(x)

#         x = layers.Dense(4096, activation='relu', name='fc2')(x)

#         x = layers.Dense(classes, activation='softmax', name='predictions')(x)

#     else:

#         if pooling == 'avg':

#             x = layers.GlobalAveragePooling2D()(x)

#         elif pooling == 'max':

#             x = layers.GlobalMaxPooling2D()(x)



#     # Ensure that the model takes into account

#     # any potential predecessors of `input_tensor`.

#     if input_tensor is not None:

#         inputs = keras_utils.get_source_inputs(input_tensor)

#     else:

#         inputs = img_input

#     # Create model.

#     model = models.Model(inputs, x, name='vgg16')



#     # Load weights.

#     if weights == 'imagenet':

#         if include_top:

#             weights_path = keras_utils.get_file(

#                 'vgg16_weights_tf_dim_ordering_tf_kernels.h5',

#                 WEIGHTS_PATH,

#                 cache_subdir='models',

#                 file_hash='64373286793e3c8b2b4e3219cbf3544b')

#         else:

#             weights_path = keras_utils.get_file(

#                 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',

#                 WEIGHTS_PATH_NO_TOP,

#                 cache_subdir='models',

#                 file_hash='6d6bbae143d832006294945121d1f1fc')

#         model.load_weights(weights_path)

#         if backend.backend() == 'theano':

#             keras_utils.convert_all_kernels_in_model(model)

#     elif weights is not None:

#         model.load_weights(weights)



#     return model

m=VGG16()
m.summary()
import keras

model=keras.applications.inception_v3.InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)

model.summary()
import keras

from keras.models import Model

from keras.layers import *

from keras.layers import Input, Dense,Dropout

import keras.layers as layers

EPOCHS = 40

BATCH_SIZE = 16

image_shape = (621,189,3)

data_dir = './data'

runs_dir = './runs'

models=None

#default input is (None, 299, 299, 3)

def conv2d_bn(x,filters,num_row,num_col,padding='same',strides=(1, 1), name=None):

    """Utility function to apply conv + BN.

    # Arguments

        x: input tensor.

        filters: filters in `Conv2D`.

        num_row: height of the convolution kernel.

        num_col: width of the convolution kernel.

        padding: padding mode in `Conv2D`.

        strides: strides in `Conv2D`.

        name: name of the ops; will become `name + '_conv'`

            for the convolution and `name + '_bn'` for the

            batch norm layer.

    # Returns

        Output tensor after applying `Conv2D` and `BatchNormalization`.

    """

    if name is not None:

        bn_name = name + '_bn'

        conv_name = name + '_conv'

    else:

        bn_name = None

        conv_name = None

    if keras.backend.image_data_format() == 'channels_first':

        bn_axis = 1

    else:

        bn_axis = 3

    x = keras.layers.Conv2D(

        filters, (num_row, num_col),

        strides=strides,

        padding=padding,

        use_bias=False,

        name=conv_name)(x)

    x = keras.layers.BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)

    x = keras.layers.Activation('relu', name=name)(x)

    return x




def model_keras():

    image_shape = (621,189,3)

    if keras.backend.image_data_format() == 'channels_first':

        channel_axis = 1

    else:

        channel_axis = 3

    img_input = keras.layers.Input(shape=image_shape)

    #image_shape = (621,189)

    #process block with convs

    x = conv2d_bn(img_input, 32, 3, 3, strides=(1, 1), padding='valid')

    x = conv2d_bn(x, 32, 3, 3, padding='valid')

    x = conv2d_bn(x, 64, 3, 3)

    x = keras.layers.MaxPooling2D((3, 3), strides=(1, 1))(x)

    x = conv2d_bn(x, 80, 1, 1, padding='valid')

    x = conv2d_bn(x, 192, 3, 3, padding='valid')

    x = keras.layers.MaxPooling2D((3, 3), strides=(1, 1))(x)

    # 609x177x192

    #mixed0: 609x177x256

    #inception+resnet block1 

    branch1x1 = conv2d_bn(x, 64, 1, 1)



    branch5x5 = conv2d_bn(x, 48, 1, 1)

    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)



    branch3x3dbl = conv2d_bn(x, 64, 1, 1)

    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)



    branch_pool = keras.layers.AveragePooling2D((3, 3),

                                          strides=(1, 1),

                                          padding='same')(x)

    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)

    x = keras.layers.concatenate(

        [branch1x1, branch5x5, branch3x3dbl, branch_pool],

        axis=channel_axis,

        name='mixed0')



    # mixed 1: 609x177x 288 BLOCK2

    branch1x1 = conv2d_bn(x, 64, 1, 1)



    branch5x5 = conv2d_bn(x, 48, 1, 1)

    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)



    branch3x3dbl = conv2d_bn(x, 64, 1, 1)

    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)



    branch_pool = keras.layers.AveragePooling2D((3, 3),

                                          strides=(1, 1),

                                          padding='same')(x)

    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)

    x = keras.layers.concatenate(

        [branch1x1, branch5x5, branch3x3dbl, branch_pool],

        axis=channel_axis,

        name='mixed1')

    

    # mixed 2: 609x177x 288 BLOCK3

    branch1x1 = conv2d_bn(x, 64, 1, 1)



    branch5x5 = conv2d_bn(x, 48, 1, 1)

    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)



    branch3x3dbl = conv2d_bn(x, 64, 1, 1)

    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)



    branch_pool = keras.layers.AveragePooling2D((3, 3),

                                          strides=(1, 1),

                                          padding='same')(x)

    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)

    x = keras.layers.concatenate(

        [branch1x1, branch5x5, branch3x3dbl, branch_pool],

        axis=channel_axis,

        name='mixed2')



    # mixed 3: 304 x 88 x 768 BLOCK4

    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')



    branch3x3dbl = conv2d_bn(x, 64, 1, 1)

    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch3x3dbl = conv2d_bn(

        branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')



    branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = layers.concatenate(

        [branch3x3, branch3x3dbl, branch_pool],

        axis=channel_axis,

        name='mixed3')



    # mixed 4: 304 x 88 x 768 BLOCK5

    branch1x1 = conv2d_bn(x, 192, 1, 1)



    branch7x7 = conv2d_bn(x, 128, 1, 1)

    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)

    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)



    branch7x7dbl = conv2d_bn(x, 128, 1, 1)

    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)

    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)

    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)

    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)



    branch_pool = layers.AveragePooling2D((3, 3),

                                          strides=(1, 1),

                                          padding='same')(x)

    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)

    x = layers.concatenate(

        [branch1x1, branch7x7, branch7x7dbl, branch_pool],

        axis=channel_axis,

        name='mixed4')

    

    # mixed 5, 6: 304 x 88 x 768 BLOCK6

    branch1x1 = conv2d_bn(x, 192, 1, 1)



    branch7x7 = conv2d_bn(x, 160, 1, 1)

    branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)

    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)



    branch7x7dbl = conv2d_bn(x, 160, 1, 1)

    branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)

    branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)

    branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)

    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)



    branch_pool = layers.AveragePooling2D(

        (3, 3), strides=(1, 1), padding='same')(x)

    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)

    x = layers.concatenate(

        [branch1x1, branch7x7, branch7x7dbl, branch_pool],

        axis=channel_axis,

        name='mixed' + str(5 + 0))



    # mixed 7: 304 x 88 x 768 BLOCK 7

    branch1x1 = conv2d_bn(x, 192, 1, 1)



    branch7x7 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)

    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)



    branch7x7dbl = conv2d_bn(x, 192, 1, 1)

    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)

    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)

    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)



    branch_pool = layers.AveragePooling2D((3, 3),

                                          strides=(1, 1),

                                          padding='same')(x)

    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)

    x = layers.concatenate(

        [branch1x1, branch7x7, branch7x7dbl, branch_pool],

        axis=channel_axis,

        name='mixed7')

    # mixed 8: 151 x 43 x 1280

    branch3x3 = conv2d_bn(x, 192, 1, 1)

    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3,

                          strides=(2, 2), padding='valid')



    branch7x7x3 = conv2d_bn(x, 192, 1, 1)

    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)

    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)

    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')



    branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = layers.concatenate(

        [branch3x3, branch7x7x3, branch_pool],

        axis=channel_axis,

        name='mixed8')

    # mixed 9: 151 x 43 x 2048

    for i in range(2):

        branch1x1 = conv2d_bn(x, 320, 1, 1)



        branch3x3 = conv2d_bn(x, 384, 1, 1)

        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)

        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)

        branch3x3 = layers.concatenate(

            [branch3x3_1, branch3x3_2],

            axis=channel_axis,

            name='mixed9_' + str(i))



        branch3x3dbl = conv2d_bn(x, 448, 1, 1)

        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)

        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)

        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)

        branch3x3dbl = layers.concatenate(

            [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)



        branch_pool = layers.AveragePooling2D(

            (3, 3), strides=(1, 1), padding='same')(x)

        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)

        x = layers.concatenate(

            [branch1x1, branch3x3, branch3x3dbl, branch_pool],

            axis=channel_axis,

            name='mixed' + str(9 + i))

    

    #     x = layers.GlobalAveragePooling2D(name='avg_pool')(x)

    if keras.backend.image_data_format() == 'channels_first':

        bn_axis = 1

    else:

        bn_axis = 3

    x=layers.Conv2DTranspose(1024,kernel_size=(4,4),strides=(2,2))(x)

    x = keras.layers.BatchNormalization(axis=bn_axis, scale=False, name='1')(x)

    x=layers.Conv2DTranspose(512,kernel_size=(4,4),strides=(2,2))(x)

    x = keras.layers.BatchNormalization(axis=bn_axis, scale=False, name='2')(x)

    x=layers.Conv2DTranspose(256,kernel_size=(8,8),strides=(1,1))(x)

#     x=layers.Conv2DTranspose(2048,kernel_size=(,128),strides=(1,1))(x)

#     x=layers.Conv2D(size=(2, 2))(x)

#     x = conv2d_bn(x, 128, 1, 1)(x)

    x=Dense(128, activation='relu')(x)

    x = keras.layers.BatchNormalization(axis=bn_axis, scale=False, name='3')(x)

    x=Dropout(0.5)(x)

    x=Dense(32, activation='relu')(x)

    x = keras.layers.BatchNormalization(axis=bn_axis, scale=False, name='4')(x)

    x=Dropout(0.5)(x)

    x=Dense(8, activation='relu')(x)

    x = keras.layers.BatchNormalization(axis=bn_axis, scale=False, name='5')(x)

    x=Dropout(0.5)(x)

    x=Dense(3, activation='relu')(x)

    model = Model(img_input, x, name='inception_v3')

    #140 x 42 x 2048

    #upsample in 3 blocks reduce channels. can upsample by resnet connections

    #inter block resnets can be reduced



    #FCN to reduce channels --?



    return model

model1= model_keras()

model1.summary()