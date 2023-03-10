import numpy as np

import pandas as pd 

import os

import cv2

import matplotlib.pyplot as plt

from matplotlib import colors

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten

from keras.utils.np_utils import to_categorical

from keras.losses import categorical_crossentropy

from keras.callbacks import ReduceLROnPlateau

from tqdm.notebook import tqdm

from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout

from tensorflow.keras.losses import MeanSquaredError

from tensorflow.keras.models import load_model
folders = ['83144',

 '149980',

 '20409',

 '139894',

 '177870',

 '126637',

 '1924',

 '194914',

 '113209',

 '138982']

paths = []

labels = []

base = '/kaggle/input/google-landmark/training'
for folder in tqdm(folders):

    temp_path = os.path.join(base, folder)

    for img in os.listdir(temp_path):

        paths.append(os.path.join(temp_path, img))

        labels.append(folder)
datagen = ImageDataGenerator(

        rescale=1./255,

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        validation_split=0.2

        #horizontal_flip=True,  # randomly flip images

        #vertical_flip=True  # randomly flip images

)



train_generator = datagen.flow_from_directory(

    directory= base,

    target_size=(256, 256),

    color_mode="rgb",

    batch_size=32,

    class_mode="categorical",

    shuffle=True,

    seed=42,

    subset='training'

)



validation_generator = datagen.flow_from_directory(

    directory= base,

    target_size=(256, 256),

    color_mode="rgb",

    batch_size=32,

    class_mode='categorical',

    subset='validation'

)
from sklearn.utils import class_weight

from tensorflow.keras import layers

from tensorflow.keras.applications import ResNet50

#class_weights = class_weight.compute_class_weight(

#               'balanced',

#                np.unique(train_generator.classes), 

#                train_generator.classes)

model = Sequential()

model.add(ResNet50(

    include_top=False, weights=None, pooling='avg', input_shape=(256, 256, 3)))

#test_model.add(layers.GlobalMaxPooling2D(name="gap"))

model.add(Flatten())

model.add(Dense(10, activation='softmax'))

from tensorflow.keras.optimizers import Adam

opt = Adam(lr=0.001, decay=1e-6,)



# https://towardsdatascience.com/keras-accuracy-metrics-8572eb479ec7 ??????top k accuracy

from tensorflow.keras.metrics import categorical_accuracy,top_k_categorical_accuracy

def top_5_accuracy(y_true, y_pred):

    return top_k_categorical_accuracy(y_true, y_pred, k=5)



model.compile(loss = "categorical_crossentropy", optimizer='adam', metrics=['acc', top_5_accuracy])

model.fit_generator(generator=train_generator, validation_data = validation_generator,

                              epochs = 5, verbose=2)#class_weight = class_weights, 
pd.Series(train_generator.classes).value_counts()
model.summary()
import tensorflow as tf

import skimage.io

from skimage.transform import resize

from imgaug import augmenters as iaa

from sklearn import preprocessing

from sklearn.preprocessing import LabelBinarizer,LabelEncoder

from tensorflow.keras import layers

from tensorflow.keras.layers import Input, Add, Dense, Dropout, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,GlobalAveragePooling2D,Concatenate, ReLU, LeakyReLU,Reshape, Lambda

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

from tensorflow.keras.optimizers import Adam,SGD

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential, load_model, Model

from tensorflow.keras.callbacks import LearningRateScheduler

from tensorflow.keras.preprocessing import image

from tensorflow.keras.utils import to_categorical

from tensorflow.keras import metrics

from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications.imagenet_utils import preprocess_input

from tensorflow.keras.initializers import glorot_uniform

import imgaug as ia

from imgaug import augmenters as iaa

from PIL import Image

import keras.backend as K
# dataset

class landmark(tf.keras.utils.Sequence):

    def __init__(self, paths, labels, batch_size, shape, shuffle = False, augment = False, class_size=10):

        self.paths, self.labels = paths, labels #??????????????????path???label?????????

        self.batch_size = batch_size # ?????????batch size

        self.shape = shape # ?????????shape

        self.shuffle = shuffle #?????????shuffle

        self.augment = augment #?????????augmentation

        self.on_epoch_end()

        self.class_size = class_size # ????????????class

    def __len__(self):

        return int(np.ceil(len(self.paths) / float(self.batch_size))) # ???????????????????????????batch size??????step numbers

    

    def __getitem__(self, idx):

        indexes = self.indexes[idx * self.batch_size : (idx+1) * self.batch_size] # ????????????batch?????????indexes



        paths = self.paths[indexes] # ???indexes????????????????????????paths

        X = np.zeros((paths.shape[0], self.shape[0], self.shape[1], self.shape[2])) # ?????????100??????????????????????????????128??? 128??? 3???

                                                                                    # ???????????????????????????100??? 128??? 128???3??????array

        # Generate data

        for i, path in enumerate(paths):

            X[i] = self.__load_image(path) #???????????????????????????????????????????????????



        y = self.labels[indexes]

            

        

        '''

        augmentation ???????????????????????????????????????

        if self.augment == True:

            seq = iaa.Sequential([

                iaa.OneOf([

                    iaa.Fliplr(0.5), # horizontal flips

                    

                    iaa.ContrastNormalization((0.75, 1.5)),

                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),

                    iaa.Multiply((0.8, 1.2), per_channel=0.2),

                    

                    iaa.Affine(rotate=0),

                    #iaa.Affine(rotate=90),

                    #iaa.Affine(rotate=180),

                    #iaa.Affine(rotate=270),

                    iaa.Fliplr(0.5),

                    #iaa.Flipud(0.5),

                ])], random_order=True)



            X = np.concatenate((X, seq.augment_images(X), seq.augment_images(X), seq.augment_images(X)), 0)

            y = np.concatenate((y, y, y, y), 0)

        '''

        return X, y

    def on_epoch_end(self):

        # ????????????

        self.indexes = np.arange(len(self.paths))

        if self.shuffle == True:

            np.random.shuffle(self.indexes)



    def __iter__(self):

        """Create a generator that iterate over the Sequence."""

        for item in (self[i] for i in range(len(self))):

            yield item

            

    def __load_image(self, path):

        image_norm = cv2.imread(path)/255.0

        im = resize(image_norm, (self.shape[0], self.shape[1], self.shape[2]), mode='reflect')

        return im
# ??????????????????Resnet????????????????????????????????????identity_block?????????????????????????????????????????????layers

# https://www.quora.com/What-is-the-identity-block-in-ResNet ??????identity block

def identity_block(X, f, filters, stage, block):

    

    conv_name_base = 'res' + str(stage) + block + '_branch' #????????????????????????base????????????????????????layer???????????????????????????????????????

    bn_name_base = 'bn' + str(stage) + block + '_branch' #??????

    

    F1, F2, F3 = filters # filter?????????????????????????????????sequential????????????convolution layer???filter????????????

    # ?????????filter???CNN??????????????????????????????????????? https://www.quora.com/What-is-a-filter-in-the-context-of-Convolutional-Neural-Networks

    # ????????????????????????CNN???????????? ???????????? https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks/

    # filter ??? kernel ???????????? https://stats.stackexchange.com/questions/154798/difference-between-kernel-and-filter-in-cnn

    

    X_shortcut = X #????????????identity block???????????????????????????????????????shortcut????????????????????????layers

        

    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)

    # conv2D???????????????????????????????????????layer?????????https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/

    # glorot_uniform: http://tensorflow.biotecan.com/python/Python_1.8/tensorflow.google.cn/api_docs/python/tf/glorot_uniform_initializer.html

    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)

    # batch normalization?????????

    # https://towardsdatascience.com/batch-normalization-in-neural-networks-1ac91516821c

    X = Activation('relu')(X)

    # activation??????????????? relu????????????activation function?????????

    # https://missinglink.ai/guides/neural-network-concepts/7-types-neural-network-activation-functions-right/

    

    # ??????????????????layer?????????????????????X??????????????????????????????????????????layer????????????X??????????????????X = layer???X??????????????????????????????layer????????????????????????

    # ?????????????????????????????????????????????x = f(x)

        

    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)

    X = Activation('relu')(X)



    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)



    # Add shortcut value to main path

    X = Add()([X_shortcut, X])

    X = Activation('relu')(X)

        

    return X
def convolutional_block(X, f, filters, stage, block, s = 2):

    # ??????????????????identity shortcut???block????????????????????????????????????????????????

    conv_name_base = 'res' + str(stage) + block + '_branch'

    bn_name_base = 'bn' + str(stage) + block + '_branch'

    

    F1, F2, F3 = filters

    

    X_shortcut = X

    

    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)

    X = Activation('relu')(X)

    

    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)

    X = Activation('relu')(X)

    

    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    

    X_shortcut = Conv2D(filters = F3, kernel_size = (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)

    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # ?????????shortcut???????????????conv2d???batchnorm?????????

    X = Add()([X_shortcut, X])

    X = Activation('relu')(X)

   

    return X
def ResNet50(input_shape = (256, 256, 3), classes = 10):

    X_input = Input(input_shape) # https://www.tensorflow.org/api_docs/python/tf/keras/Input

    # ???????????????layer???????????????????????????input???output

    

    X = ZeroPadding2D((3, 3))(X_input)

    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding2D

    # ????????????padding?????????????????????convolution??????????????????????????????filter size

    

    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)

    # Stride is the number of pixels shifts over the input matrix. 

    # https://medium.com/@RaghavPrabhu/understanding-of-convolutional-neural-network-cnn-deep-learning-99760835f148

    

    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)

    X = Activation('relu')(X)

    

    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/

    # ?????????????????????overfit???????????????????????????drop out

    

    # ??????????????????resnet????????????resnet??????????????????????????????????????????????????????????????????????????????accuracy?????????????????????imagenet???

    # ?????????????????????????????????????????????

    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)

    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')

    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    

    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2)

    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')

    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')

    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    

    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)

    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')

    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')

    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')

    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')

    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    

    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)

    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')

    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    

    X = AveragePooling2D(pool_size=(2, 2),name='avg_pool')(X)

    X = Flatten()(X)

    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)

    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model
model = ResNet50(input_shape = (256, 256, 3), classes = 10)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc',top_5_accuracy])

model.summary()
# ????????????????????????????????????????????????????????????path????????????label

display(len(paths))

display(len(labels))

paths = np.array(paths)

labels = np.array(labels)
# ?????????one-hot encoding labels

from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(sparse = False)

labels = np.reshape(labels, (-1, 1))

enc.fit(labels)

labels = enc.transform(labels)
labels.shape
from sklearn.model_selection import train_test_split

pathsTrain, pathsVal, labelsTrain, labelsVal = train_test_split(paths, labels, test_size = 0.2)



# ???paths ??? labels????????????????????????????????????sklearn train test split
train_generator = landmark(pathsTrain, labelsTrain, batch_size = 32, shape = (256, 256, 3), shuffle = True)

val_generator = landmark(pathsVal, labelsVal, batch_size = 32, shape = (256, 256, 3), shuffle = False)



# ???????????????train??? validation generator
model.fit_generator(

    train_generator,

    steps_per_epoch=len(train_generator),

    validation_data=val_generator,

    validation_steps=64,

    #class_weight = class_weights,

    epochs=5,

    #callbacks = [clr],

    #workers=workers,

    verbose=2)