
from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
import tensorflow as tf 
import numpy as np
import math
import matplotlib.pyplot as plt
# import os
# import argparse
def get_generator(inputs, image_size):
    """
    stack of BN - ReLU - Conv2DTranspose to generate fake image
    using Conv2DTranspose: to upsampling the image 
    
    Argumments:
        inputs: is the z-vector,
        image_size: the target image
        
    Returns: 
        generator (Model)
    """
    image_resize = image_size // 4
    kernel_size =5
    layers_filters = [128, 64, 32, 1]
    #
    x = Dense(image_size * image_resize * layers_filters[0])(inputs)
    x = Reshape((image_size, image_size, layer_filters[0]))(x)
    #
    for filters in layer_fliters:
        if filters > layer_filters[-2]:
            strides = 2
        else:
            strides = 1
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(filters = filters, kernel_size = kernel_size, strides = strides, padding = 'same')(x)
    outputs = Activation('sigmoid')(x)
    #
    generator = Model(inputs, outputs, name='generator')
    return generator
def get_discriminator(inputs):
    """
    the discriminator is similar binary CNN classifiers. 
    which the input ia a image 28 x28 x 1. that is classified either real(1.0)
    and fake(0.0)
    """
    kernel_size = 5
    layer_filters = [32, 64,128, 256]
    #
    x = inputs
def build_and_train_models():
    # load data
    (x_train, _),(_,_) = mnist.load_data()
    image_size = x_train.shape[1]
    x_train = x_train[..., tf.newaxis].astype("float32")/255.
    kernal_size = 4
    #
def train(models, x_train, params):
    return True



