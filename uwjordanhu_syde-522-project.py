# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



# Any results you write to the current directory are saved as output.
import keras

from keras import optimizers

from keras.preprocessing import image

from keras.engine import Layer

from keras.layers import Conv2D, Conv3D, UpSampling2D, InputLayer, Conv2DTranspose, Input, Reshape, merge, concatenate

from keras.layers import Activation, Dense, Dropout, Flatten

from keras.layers.normalization import BatchNormalization

from keras.callbacks import TensorBoard

from keras.models import Sequential, Model

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb

from skimage.transform import resize, rescale

from skimage.io import imsave, imread

from time import time

import numpy as np

import os

import random

import tensorflow as tf

from PIL import Image, ImageFile

import matplotlib.pyplot as plt
data = pd.read_csv('/kaggle/input/celeba-dataset/list_eval_partition.csv')

data['partition'].value_counts().sort_index()

data_dir = '/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba'

data['train_file'] = data.image_id.apply(lambda x: data_dir + '/{0}'.format(x))

data.head()
# reading images

from skimage.io import imread

import matplotlib.pyplot as plt



%matplotlib inline





plt.figure(figsize=(15, 4))

for idx, (_, entry) in enumerate(data.sample(n=5).iterrows()):

    img = imread(entry.train_file)

    print(img.shape)

    plt.subplot(1, 5, idx+1)

    plt.imshow(imread(entry.train_file))

    plt.axis('off')

    plt.title(entry.image_id)
## constants

num_train_samples = 95

num_val_samples = 10

img_height = 224

img_width = 224

batch_size = 4

epochs = 100
## Custom data generator



class colorizationGenerator():

  

  def __init__(self, path, files, batch_size):

    self.path = path

    self.files = files

    self.batch_size = batch_size



  def get_input(self, path, filename):

    img = imread(os.path.join(path,filename))

    return(img)



  def get_output(self, path, filename):

    img = imread(os.path.join(path,filename))

    img = resize(img,(224,224,3))

    labImage = rgb2lab(img)

    abch = labImage[:,:,1:]/128.0

    return np.array(abch)



  def preprocess_input(self, img):

    img = resize(img,(224,224,3))

    labImage = rgb2lab(img)

    lch = labImage[:,:,0]/100.0

    l3ch = gray2rgb(lch)

    return np.array(l3ch)



  def image_generator(self):

      while True:

            # Select files (paths/indices) for the batch

            batch_paths  = np.random.choice(a    = self.files, 

                                            size = self.batch_size)

            batch_input  = []

            batch_output = [] 

            

            # Read in each input, perform preprocessing and get labels

            for input_path in batch_paths:

                currinput = self.get_input(self.path, input_path)

                output = self.get_output(self.path, input_path)

              

                currinput = self.preprocess_input(currinput)

                batch_input += [currinput]

                batch_output += [output]

                

            # Return a tuple of (input, output) to feed the network

            batch_x = np.array(batch_input)

            batch_y = np.array(batch_output)

          

            yield(batch_x, batch_y)
## The model

modelInput = Input(shape=(img_height, img_width, 3))

vggModel = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_tensor=modelInput)



output = vggModel.layers[-1].output

vgg16TruncModel = Model(vggModel.input, output)

for layer in vgg16TruncModel.layers:

  layer.trainable=False



decoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(vgg16TruncModel.output)

# decoder_output = BatchNormalization()(decoder_output)

decoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(decoder_output)

# decoder_output = BatchNormalization()(decoder_output)

decoder_output = UpSampling2D((2, 2))(decoder_output)

decoder_output = Conv2D(64, (3,3), activation='relu', padding='same')(decoder_output)

# decoder_output = BatchNormalization()(decoder_output)

decoder_output = UpSampling2D((2, 2))(decoder_output)

decoder_output = Conv2D(32, (3,3), activation='relu', padding='same')(decoder_output)

# decoder_output = BatchNormalization()(decoder_output)

decoder_output = UpSampling2D((2, 2))(decoder_output)

decoder_output = Conv2D(16, (3,3), activation='relu', padding='same')(decoder_output)

# decoder_output = BatchNormalization()(decoder_output)

decoder_output = UpSampling2D((2, 2))(decoder_output)

decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output)

decoder_output = UpSampling2D((2, 2))(decoder_output)

model = Model(inputs=modelInput, outputs=decoder_output)
model.summary()
model.compile(optimizer='adam', loss='mse' , metrics=['accuracy'])
dataGenerator = colorizationGenerator(data_dir,

                                      os.listdir(data_dir), batch_size)



from keras.callbacks import ModelCheckpoint



checkpoint_callback = ModelCheckpoint('/best_model_final.h5', monitor='loss', verbose=1, save_best_only=True, mode='min')
test_img = imread(data_dir + '/000001.jpg')

test_img = resize(test_img,(224,224,3))

lab = rgb2lab(test_img)

print(lab.shape)

l = lab[:,:,0]

a = lab[:,:,1]

b = lab[:,:,2]

print(np.max(l))

print(np.max(a))

print(np.max(b))

plt.imshow(test_img)
images = next(dataGenerator.image_generator())

print(images[1][0].shape)

print(np.max(images[0][0][0]))

print(np.max(images[1][0][:,:,0]))

print(np.max(images[1][0][:,:,1]))

cur = np.zeros((224, 224, 3))

cur[:,:,0] = images[0][0][:,:,0]*100.0

cur[:,:,1:] = images[1][0]*128.0

output = lab2rgb(cur)

plt.imshow(output)

print(np.max(cur[:,:,0]))

print(np.max(cur[:,:,1]))

print(np.max(cur[:,:,2]))
history = model.fit_generator(

    dataGenerator.image_generator(),

    steps_per_epoch=num_train_samples // batch_size,

    validation_data=valGenerator.image_generator(),

    validation_steps=1,

    epochs=epochs,

    callbacks=[checkpoint_callback]

)
acc = history.history['accuracy']

mse = history.history['loss']



epochs_range = range(len(acc))



plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)

plt.plot(epochs_range, acc, label='Training Accuracy')

plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.legend(loc='lower right')

plt.title('Training Accuracy')



plt.subplot(1, 2, 2)

plt.plot(epochs_range, mse, label='Training MSE')

plt.plot(history.history['val_loss'], label='Validation MSE')

plt.legend(loc='upper right')

plt.title('Training MSE')



plt.show()