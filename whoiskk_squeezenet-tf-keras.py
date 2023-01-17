import numpy as np

import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Layer
from tensorflow.keras.layers import Concatenate, Add
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, Dropout

from tensorflow.keras.utils import plot_model

import warnings
warnings.simplefilter('ignore')
one = (1, 1)
two = (2, 2)
three = (3, 3)
five = (5, 5)
seven = (7, 7)
thirteen = (13, 13)

input_shape = (224, 224, 3)
class FireModule(object):
    """
    Fire Module computed as per the SqueezeNet paper
    """
    
    def __init__(self, layer_number: int, activation: str, kernel_initializer: str) -> None:
        """
        Constructor
        
        Arguments:
          layer_number       : Index of the Fire Module
          activation         : Activation to be used
          kernel_initializer : Kernel Weight Initialization technique
          
        Returns:
          None
        """
        
        self.layer_number = layer_number
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        
    def build_module(self, fire_input: Layer) -> Layer:
        """
        Build the SqueezeNet
        
        Arguments:
          fire_input       : Input to Fire Module
          
        Returns:
          model            : SqueezeNet
        """
        
        global one, three, five
        
        output_size = 128 * (1 + (self.layer_number//2))
        
        squeeze_1x1_filters = 16 * (1 + (self.layer_number//2))
        expand_1x1_filters = expand_3x3_filters = output_size//2

        squeeze_1x1 = Conv2D(name=f'fire_{self.layer_number+2}_squeeze_1x1',
            filters=squeeze_1x1_filters, kernel_size=one, strides=1, padding='valid', activation=self.activation, 
            kernel_initializer=self.kernel_initializer)(fire_input)
        expand_1x1 = Conv2D(name=f'fire_{self.layer_number+2}_expand_1x1',
            filters=expand_1x1_filters, kernel_size=one, strides=1, padding='valid', activation=self.activation, 
            kernel_initializer=self.kernel_initializer)(squeeze_1x1)
        expand_3x3 = Conv2D(name=f'fire_{self.layer_number+2}_expand_3x3',
            filters=expand_3x3_filters, kernel_size=three, strides=1, padding='same', activation=self.activation, 
            kernel_initializer=self.kernel_initializer)(squeeze_1x1)

        fire = Concatenate(name=f'fire_{self.layer_number+2}')([expand_1x1, expand_3x3])
        
        return fire
class SqueezeNet(object):
    """
    SqueezeNet Architecture
    """
    
    def __init__(self, activation: str='relu', kernel_initializer: str='glorot_uniform') -> None:
        """
        Constructor
        
        Arguments:
          activation         : Activation to be used
          kernel_initializer : Kernel Weight Initialization technique
          
        Returns:
          None
        """
        
        self.activation = activation
        self.kernel_initializer = kernel_initializer
    
    def vanilla_model(self, input_shape: tuple=(224, 224, 3), n_classes: int=1000) -> None:
        """
        Vanilla Implementation of SqueezeNet
        
        Arguments:
          input_shape         : Input Shape of the images
          n_classes           : Number of output classes
          
        Returns:
          None
        """
        
        inp = Input(shape=input_shape, name='Input')
        
        # Conv1 Layer
        conv_1 = Conv2D(name="Conv_1",
            filters=96, kernel_size=seven, strides=2, padding='same', activation=self.activation, kernel_initializer=self.kernel_initializer)(inp)
        maxpool_1 = MaxPool2D(name="MaxPool_1",
            pool_size=three, strides=2)(conv_1)
        
        # Fire 2-4
        fire_2 = FireModule(layer_number=0, activation=self.activation, kernel_initializer=self.kernel_initializer).build_module(maxpool_1)
        fire_3 = FireModule(layer_number=1, activation=self.activation, kernel_initializer=self.kernel_initializer).build_module(fire_2)
        fire_4 = FireModule(layer_number=2, activation=self.activation, kernel_initializer=self.kernel_initializer).build_module(fire_3)
        
        # Max Pool after Fire4 Module
        maxpool_2 = MaxPool2D(name="MaxPool_2",
            pool_size=three, strides=2)(fire_4)
        
        # Fire 5-8
        fire_5 = FireModule(layer_number=3, activation=self.activation, kernel_initializer=self.kernel_initializer).build_module(maxpool_2)
        fire_6 = FireModule(layer_number=4, activation=self.activation, kernel_initializer=self.kernel_initializer).build_module(fire_5)
        fire_7 = FireModule(layer_number=5, activation=self.activation, kernel_initializer=self.kernel_initializer).build_module(fire_6)
        fire_8 = FireModule(layer_number=6, activation=self.activation, kernel_initializer=self.kernel_initializer).build_module(fire_7)

        # Max Pool after Fire8 Module
        maxpool_3 = MaxPool2D(name="MaxPool_3",
            pool_size=three, strides=2)(fire_8)
        
        fire_9 = FireModule(layer_number=7, activation=self.activation, kernel_initializer=self.kernel_initializer).build_module(maxpool_3)
        
        # Dropout
        dropout = Dropout(0.5, name="Dropout")(fire_9)
        
        # Conv10 layer
        conv_10 = Conv2D(name="Conv_10",
            filters=1000, kernel_size=one, strides=1, padding='valid', activation=self.activation, kernel_initializer=self.kernel_initializer)(dropout)
        gap_11 = GlobalAveragePooling2D()(conv_10)

        
        if n_classes != 1000:
            # Add Dense(n_classes) and ouput == Dense layer
            out = Dense(n_classes, activation='softmax')(gap_11)
        else:
            out = gap_11
        
        self.model = Model(inputs=inp, outputs=out)
        
        
    def bypass_model(self, input_shape: tuple=(224, 224, 3), n_classes: int=1000) -> None:
        """
        Residual Inspired Bypass Implementation of SqueezeNet
        
        Arguments:
          input_shape         : Input Shape of the images
          n_classes           : Number of output classes
          
        Returns:
          None
        """
        
        inp = Input(shape=input_shape, name='Input')
        
        # Conv1 Layer
        conv_1 = Conv2D(name="Conv_1",
            filters=96, kernel_size=seven, strides=2, padding='same', activation=self.activation, kernel_initializer=self.kernel_initializer)(inp)
        maxpool_1 = MaxPool2D(name="MaxPool_1",
            pool_size=three, strides=2)(conv_1)
        
        # Fire 2-4
        fire_2 = FireModule(layer_number=0, activation=self.activation, kernel_initializer=self.kernel_initializer).build_module(maxpool_1)
        fire_3 = FireModule(layer_number=1, activation=self.activation, kernel_initializer=self.kernel_initializer).build_module(fire_2)
        bypass_1 = Add(name="Bypass_1")([fire_2, fire_3])
        fire_4 = FireModule(layer_number=2, activation=self.activation, kernel_initializer=self.kernel_initializer).build_module(bypass_1)
        
        # Max Pool after Fire4 Module
        maxpool_2 = MaxPool2D(name="MaxPool_2",
            pool_size=three, strides=2)(fire_4)
        
        # Fire 5-8
        fire_5 = FireModule(layer_number=3, activation=self.activation, kernel_initializer=self.kernel_initializer).build_module(maxpool_2)
        bypass_2 = Add(name="Bypass_2")([maxpool_2, fire_5])
        fire_6 = FireModule(layer_number=4, activation=self.activation, kernel_initializer=self.kernel_initializer).build_module(bypass_2)
        fire_7 = FireModule(layer_number=5, activation=self.activation, kernel_initializer=self.kernel_initializer).build_module(fire_6)
        bypass_3 = Add(name="Bypass_3")([fire_6, fire_7])
        fire_8 = FireModule(layer_number=6, activation=self.activation, kernel_initializer=self.kernel_initializer).build_module(bypass_3)

        # Max Pool after Fire8 Module
        maxpool_3 = MaxPool2D(name="MaxPool_3",
            pool_size=three, strides=2)(fire_8)
        
        fire_9 = FireModule(layer_number=7, activation=self.activation, kernel_initializer=self.kernel_initializer).build_module(maxpool_3)
        bypass_4 = Add(name="Bypass_4")([maxpool_3, fire_9])
        
        # Dropout
        dropout = Dropout(0.5, name="Dropout")(bypass_4)
        
        # Conv10 layer
        conv_10 = Conv2D(name="Conv_10",
            filters=1000, kernel_size=one, strides=1, padding='valid', activation=self.activation, kernel_initializer=self.kernel_initializer)(dropout)
        gap_11 = GlobalAveragePooling2D()(conv_10)

        
        if n_classes != 1000:
            out = Dense(n_classes, activation='softmax')(gap_11)
        else:
            out = gap_11
        
        self.model = Model(inputs=inp, outputs=out)
    
    
    def build_model(self, input_shape: tuple=(224, 224, 3), n_classes: int=1000, choice: str='vanilla') -> Model:
        """
        Build SqueezeNet
        
        Arguments:
          input_shape         : Input Shape of the images
          n_classes           : Number of output classes
          choice              : Type of architecture (vanilla/bypass)
        Returns:
          model               : SqueezeNet Model
        """
        
        if choice == "vanilla":
            self.vanilla_model(input_shape, n_classes)
        else:
            self.bypass_model(input_shape, n_classes)
        
        return self.model
snet = SqueezeNet()

model = snet.build_model(n_classes=10, choice='bypass')
model.summary()
plot_model(model, show_shapes=True, show_layer_names=True)
