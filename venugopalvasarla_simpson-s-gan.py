# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.



# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        (os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import os

import cv2

from IPython.display import Image, display

path = '../input/simpsons-faces/cropped/'

image_paths = os.listdir(path)

i = 0

print('lenght of all the data available:',len(image_paths))

for image_path in image_paths:

    display(Image(path+image_path))

    img = cv2.imread(path+image_path)

    print(img.shape)

    i = i+1

    if i == 5:

        break
#Lets resize all the data into 64*64*3

input_images = []

for image_path in image_paths:

    image = cv2.imread(path+image_path)

    image_resized = cv2.resize(image, (32,32))

    input_images.append(np.array(image_resized))
#the discriminator:

def Discriminator():

    in_layer_1 = Dense(3072, activation = 'sigmoid')(in_2)

    layer_2 = Reshape((2,3072,1))(concatenate([in_1, in_layer_1]))

    conv_layer = Conv2D(filters = 1, kernel_size = (2,1), use_bias = False,

                       name = 'conv_layer')(layer_2)

    out_layer = Flatten()(conv_layer)

    return out_layer
from keras.models import Model

from keras.layers import Dense, Conv2D, Reshape, Input, concatenate, Flatten

in_1 = Input((3072,))

in_2 = Input((9877,))

model = Discriminator()

discriminator_model = Model([in_1,in_2], model)

discriminator_model.get_layer('conv_layer').trainable = False

discriminator_model.get_layer('conv_layer').set_weights = ([np.array([[[[-1.0]]], [[[1.0]]]])])

print('the discriminator model is given: \n',discriminator_model.summary())
#now let's build our generator

noise_shape = 9877

def Generator(noise = (noise_shape,)):

    in_3 = Input(noise)

    dense_2 = Dense(3072, activation = 'linear')(in_3)

    model = Model(inputs = in_3, outputs = [dense_2, Reshape((9877,))(in_3)])

    print(model.summary())

    return model
generator_model = Generator(noise=(noise_shape,))
input_images = np.array(input_images)

print('the shape of input images is:', input_images.shape)
#considering our input_images as the output of the discriminator.

#i.e. y_train data is the real images

#setting the y train data

y_train = input_images[:input_images.shape[0],:,:,:]/255

#reshaping the images into the data of 64*64*3 columns

y_train = y_train.reshape((-1, 3072))
#creating a input data

#X train will be a square diagonal of size of train samples

X_train  = np.zeros((9877, 9877))

for  i in range(9877):

    X_train[i,i] = 1

#second input for the discriminator must be null at training.

#so lets take all values zeros

X_train_2 = np.zeros((9877, 3072))
learning_rate = 0.5

from keras.callbacks import LearningRateScheduler

discriminator_model.compile(optimizer='adam', loss='binary_crossentropy')

for k in range(20):

    lr_scheduler = LearningRateScheduler(lambda x: learning_rate)

    history = discriminator_model.fit([X_train_2, X_train], y_train, epochs = 10, batch_size = 32,

                                     callbacks = [lr_scheduler], verbose = 0)

    print('Epoch',(k+1)*10,'/200 - loss =',history.history['loss'][-1] )

    if history.history['loss'][-1]<0.533: lr = 0.1
print('Discriminator Recalls from Memory')  

import matplotlib.pyplot as plt

from PIL import Image

for k in range(5):

    plt.figure(figsize=(15,3))

    for j in range(0,5):

        xx = np.zeros((9877))

        xx[np.random.randint(9877)] = 1

        plt.subplot(1,5,j+1)

        img = discriminator_model.predict([X_train_2[0,:].reshape((-1, 3072)),xx.reshape((-1,9877))]).reshape((-1,32,32,3))

        img = Image.fromarray((255*img).astype('uint8').reshape((32,32,3)))

        plt.imshow(img)

    plt.show()
k = discriminator_model.predict([X_train_2[0,:].reshape((-1, 3072)),xx.reshape((-1,9877))]).reshape((-1,32,32,3))
y_train[1]*255
k*255