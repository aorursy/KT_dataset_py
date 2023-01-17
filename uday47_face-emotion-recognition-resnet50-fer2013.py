import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from keras import layers

from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D

from keras.models import Model, load_model

from keras.preprocessing import image

from keras.utils import layer_utils

from keras.utils.data_utils import get_file

from keras.applications.imagenet_utils import preprocess_input

import pydot

from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot

from keras.utils import plot_model



from keras.initializers import glorot_uniform

import scipy.misc

from matplotlib.pyplot import imshow

%matplotlib inline



import keras.backend as K

K.set_image_data_format('channels_last')

K.set_learning_phase(1)
os.mkdir("./model")
def identity_block(X, f, filters, stage, block):

    """

    Implementation of the identity block

    

    Arguments:

    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)

    f -- integer, specifying the shape of the middle CONV's window for the main path

    filters -- python list of integers, defining the number of filters in the CONV layers of the main path

    stage -- integer, used to name the layers, depending on their position in the network

    block -- string/character, used to name the layers, depending on their position in the network

    

    Returns:

    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)

    """

    

    # defining name basis

    conv_name_base = 'res' + str(stage) + block + '_branch'

    bn_name_base = 'bn' + str(stage) + block + '_branch'

    

    # Retrieve Filters

    F1, F2, F3 = filters

    

    # Save the input value. You'll need this later to add back to the main path. 

    X_shortcut = X

    

    # First component of main path

    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)

    X = Activation('relu')(X)

    

   

    # Second component of main path

    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)

    X = Activation('relu')(X)



    # Third component of main path 

    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)



    # Final step: Add shortcut value to main path, and pass it through a RELU activation 

    X = Add()([X_shortcut,X])

    X = Activation("relu")(X)

        

    return X
def convolutional_block(X, f, filters, stage, block, s = 2):

    """

    Implementation of the convolutional block4

    

    Arguments:

    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)

    f -- integer, specifying the shape of the middle CONV's window for the main path

    filters -- python list of integers, defining the number of filters in the CONV layers of the main path

    stage -- integer, used to name the layers, depending on their position in the network

    block -- string/character, used to name the layers, depending on their position in the network

    s -- Integer, specifying the stride to be used

    

    Returns:

    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)

    """

    

    # defining name basis

    conv_name_base = 'res' + str(stage) + block + '_branch'

    bn_name_base = 'bn' + str(stage) + block + '_branch'

    

    # Retrieve Filters

    F1, F2, F3 = filters

    

    # Save the input value

    X_shortcut = X





    ##### MAIN PATH #####

    # First component of main path 

    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a',padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)

    X = Activation('relu')(X)

    

    # Second component of main path 

    X = Conv2D(F2, (f, f), strides = (1,1), name = conv_name_base + '2b',padding = 'same', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)

    X = Activation('relu')(X)



    # Third component of main path 

    X = Conv2D(F3, (1, 1), strides = (1,1), name = conv_name_base + '2c',padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)



    ##### SHORTCUT PATH #### 

    X_shortcut = Conv2D(F3, (1, 1), strides = (s,s), name = conv_name_base + '1',padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)

    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)



    # Final step: Add shortcut value to main path, and pass it through a RELU activation 

    X = Add()([X_shortcut,X])

    X = Activation("relu")(X)

        

    return X
def Net50(input_shape = (48,48,1), classes = 7):

    """

    Implementation of the popular ResNet50 the following architecture:

    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3

    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER



    Arguments:

    input_shape -- shape of the images of the dataset

    classes -- integer, number of classes



    Returns:

    model -- a Model() instance in Keras

    """

    

    # Define the input as a tensor with shape input_shape

    X_input = Input(input_shape)



    

    # Zero-Padding

    #X = ZeroPadding2D((1, 1))(X_input)

    X = X_input

    # Stage 1



    X = Conv2D(8, (3, 3), strides = (1, 1), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)

    X = Activation('relu')(X)

    # removed maxpool

    #X = MaxPooling2D((3, 3), strides=(2, 2))(X)



    # Stage 2

    X = convolutional_block(X, f = 3, filters = [32, 32, 128], stage = 2, block='a', s = 1)

    X = identity_block(X, 3, [32, 32, 128], stage=2, block='b')

    X = identity_block(X, 3, [32, 32, 128], stage=2, block='c')





    # Stage 3 

    X = convolutional_block(X, f = 3, filters = [64,64,256], stage = 3, block='a', s = 2)

    X = identity_block(X, 3, [64,64,256], stage=3, block='b')

    X = identity_block(X, 3, [64,64,256], stage=3, block='c')

    X = identity_block(X, 3, [64,64,256], stage=3, block='d')



    # Stage 4 

    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 4, block='a', s = 2)

    X = identity_block(X, 3, [128, 128, 512], stage=4, block='b')

    X = identity_block(X, 3, [128, 128, 512], stage=4, block='c')

    X = identity_block(X, 3, [128, 128, 512], stage=4, block='d')

    X = identity_block(X, 3, [128, 128, 512], stage=4, block='e')

    X = identity_block(X, 3, [128, 128, 512], stage=4, block='f')



    # Stage 5 

    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 5, block='a', s = 2)

    X = identity_block(X, 3, [256, 256, 1024], stage=5, block='b')

    X = identity_block(X, 3, [256, 256, 1024], stage=5, block='c')



    # AVGPOOL . 

    X = AveragePooling2D((2,2), name='avg_pool')(X)

    

    # output layer

    X = Flatten()(X)

    X = Dense(512, activation = 'relu', name='fc1024' , kernel_initializer = glorot_uniform(seed=0))(X)

    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)

    

    

    # Create model

    model = Model(inputs = X_input, outputs = X, name='Net50')



    return model
import cv2

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping

from keras.callbacks import ReduceLROnPlateau

import tensorflow as tf

 

dataset_path = '../input/fer2013/fer2013.csv'

image_size=(48,48)

batch_size = 32



def load_fer2013():

    data = pd.read_csv(dataset_path)

    data = (data[data['pixels'].notnull()])

    pixels = data['pixels'].tolist()

    width, height = 48, 48

    faces = []

    for pixel_sequence in pixels:

        face = [int(pixel) for pixel in pixel_sequence.split(' ')]

        face = np.asarray(face).reshape(width, height)

        face = cv2.resize(face.astype('uint8'),image_size)

        faces.append(face.astype('float32'))

    faces = np.asarray(faces)

    faces = np.expand_dims(faces, -1)

    emotions = (data['emotion'])#.values

    return faces, emotions

 

def preprocess_input(x, v2=True):

    x = x.astype('float32')

    x = x / 255.0

    if v2:

        x = x - 0.5

        x = x * 2.0

    return x

 

faces, emotions = load_fer2013()

faces = preprocess_input(faces)

xtrain, xtest,ytrain,ytest = train_test_split(faces, emotions,test_size=0.2,shuffle=True)

#Data Augumentation

data_generator = ImageDataGenerator(

                        featurewise_center=False,

                        featurewise_std_normalization=False,

                        rotation_range=10,

                        shear_range = 10,

                        width_shift_range=0.1,

                        height_shift_range=0.1,

                        zoom_range=.1,

                        horizontal_flip=True)

batch_size = 32

num_epochs = 150

input_shape = (48, 48, 1)

verbose = 1

num_classes = 7

patience = 25

base_path = './model/'

l2_regularization=0.01



# callbacks

log_file_path = base_path + '_emotion_training.log'

csv_logger = CSVLogger(log_file_path, append=False)

early_stop = EarlyStopping('val_loss', patience=patience)

reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience/2), verbose=1)

trained_models_path = base_path + '_mini_XCEPTION'

model_names = trained_models_path + '.{epoch:02d}-{val_accuracy:.2f}.hdf5'

model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,save_best_only=True)

callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

from keras.optimizers import Adam, SGD

model = Net50(input_shape = (48, 48, 1), classes = 7)

optimizer = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-7)

model.compile(optimizer= optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(data_generator.flow(xtrain, ytrain,),

                        steps_per_epoch=len(xtrain) / batch_size,

                        epochs=num_epochs, verbose=1, callbacks=callbacks,

                        validation_data= (xtest,ytest))
import keras

from matplotlib import pyplot as plt

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()



plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()
testloss = model.evaluate(xtest, ytest) 

print("Test Loss " + str(testloss[0]))

print("Test Acc: " + str(testloss[1]))

trainloss = model.evaluate(xtrain, ytrain) 

print("Train Loss " + str(trainloss[0]))

print("Train Acc: " + str(trainloss[1]))
plot_model(model, to_file='model.png')

SVG(model_to_dot(model).create(prog='dot', format='svg'))