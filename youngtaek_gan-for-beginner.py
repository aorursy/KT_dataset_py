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

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from keras.layers import Input, Dense, Reshape, Flatten, Dropout

from keras.layers import BatchNormalization, Activation, ZeroPadding2D

from keras.layers.advanced_activations import LeakyReLU

from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose

from keras.models import Sequential, Model

from keras.optimizers import Adam,SGD

from random import randint

import keras

from keras import layers

import matplotlib.pyplot as plt

import sys

import numpy as np

from keras.preprocessing import image



print(os.listdir("../input"))
from os import listdir, makedirs

from os.path import join, exists, expanduser



cache_dir = expanduser(join('~', '.keras'))

if not exists(cache_dir):

    makedirs(cache_dir)

datasets_dir = join(cache_dir, 'datasets') # /cifar-10-batches-py

if not exists(datasets_dir):

    makedirs(datasets_dir)





!cp /kaggle/input/cifar-10-python.tar.gz ~/.keras/datasets/

!ln -s  ~/.keras/datasets/cifar-10-python.tar.gz ~/.keras/datasets/cifar-10-batches-py.tar.gz

!tar xzvf ~/.keras/datasets/cifar-10-python.tar.gz -C ~/.keras/datasets/
class Generator(object):

    def __init__(self, width = 32, height= 32, channels = 3, latent_size=100):

        self.W = width

        self.H = height

        self.C = channels

        self.OPTIMIZER = Adam(lr=1e-4, beta_1=0.2)



        self.LATENT_SPACE_SIZE = latent_size

        self.latent_space = np.random.normal(0,1,(self.LATENT_SPACE_SIZE,))



        self.Generator = self.model()

        self.Generator.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZER)

        self.summary()



    def model(self):

        model = Sequential()

        model.add(Dense(128 * 16 * 16, input_shape=(self.LATENT_SPACE_SIZE,)))

        model.add(LeakyReLU(alpha=0.2))

        model.add(BatchNormalization())

        

        model.add(Reshape((16, 16, 128)))

        

        model.add(Conv2D(256, (5,5), padding='same'))

        model.add(LeakyReLU(alpha=0.2))

        model.add(BatchNormalization())

        

        model.add(Conv2DTranspose(256, (4,4), strides=2, padding='same'))

        model.add(LeakyReLU(alpha=0.2))

        model.add(BatchNormalization())

        

        model.add(Conv2D(256, (5,5), padding='same'))

        model.add(LeakyReLU(alpha=0.2))

        model.add(BatchNormalization())

        

        model.add(Conv2D(256, (5,5), padding='same'))

        model.add(LeakyReLU(alpha=0.2))

        model.add(BatchNormalization())

        

        model.add(Conv2D(self.C, (7,7), activation='tanh', padding='same'))

        

        return model



    def summary(self):

        return self.Generator.summary()

    

class Discriminator(object):

    def __init__(self, width = 32, height= 32, channels = 3, latent_size=100):

        self.CAPACITY = width*height*channels

        self.SHAPE = (width,height,channels)

        self.OPTIMIZER = Adam(lr=1e-4, beta_1=0.2)





        self.Discriminator = self.model()

        self.Discriminator.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZER, metrics=['accuracy'] )

        self.summary()



    def model(self):

        model = Sequential()

        model.add(Conv2D(128, (3,3), input_shape=((32, 32, 3))))

        model.add(LeakyReLU(alpha=0.2))

        model.add(BatchNormalization())

        

        model.add(Conv2D(128, (4,4), strides=2))

        model.add(LeakyReLU(alpha=0.2))

        model.add(BatchNormalization())

        

        model.add(Conv2D(128, (4,4), strides=2))

        model.add(LeakyReLU(alpha=0.2))

        model.add(BatchNormalization())

        

        model.add(Conv2D(128, (4,4), strides=2))

        model.add(LeakyReLU(alpha=0.2))

        model.add(BatchNormalization())

        

        model.add(Flatten())

        model.add(Dropout(0.4))

        

        model.add(Dense(1, activation='sigmoid'))

        return model



    def summary(self):

        return self.Discriminator.summary()
class GAN(object):

    def __init__(self,discriminator,generator):

        self.OPTIMIZER = SGD(lr=2e-4,nesterov=True)

        

        self.Generator = generator



        self.Discriminator = discriminator

        self.Discriminator.trainable = False

        

        self.gan_model = self.model()

        self.gan_model.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZER)

        self.summary()



    def model(self):

        model = Sequential()

        model.add(self.Generator)

        model.add(self.Discriminator)

        return model



    def summary(self):

        return self.gan_model.summary()
class Trainer:

    def __init__(self, width = 32, height= 32, channels = 3, latent_size=100, epochs =50000, batch=32, checkpoint=50, model_type=3):

        self.W = width

        self.H = height

        self.C = channels

        self.EPOCHS = epochs

        self.BATCH = batch

        self.CHECKPOINT = checkpoint

        self.model_type=model_type



        self.LATENT_SPACE_SIZE = latent_size



        self.generator = Generator(height=self.H, width=self.W, channels=self.C, latent_size=self.LATENT_SPACE_SIZE)

        self.discriminator = Discriminator(height=self.H, width=self.W, channels=self.C)

        self.gan = GAN(generator=self.generator.Generator, discriminator=self.discriminator.Discriminator)



        self.load_cifar10()



    def load_cifar10(self,model_type=3):

        (X_train, y_train), (_, _) = keras.datasets.cifar10.load_data()

        X_train = X_train[y_train.flatten() == model_type]

        X_train = np.asarray(X_train, dtype='float32') / 255.0

        X_train = (X_train - 0.5) / 0.5

        self.X_train = X_train        

        return

        



    def train(self):

        for e in range(self.EPOCHS):

            # Train Discriminator

            # Make the training batch for this model be half real, half noise

            # Grab Real Images for this training batch

            count_real_images = int(self.BATCH/2)

            starting_index = randint(0, (len(self.X_train)-count_real_images))

            real_images_raw = self.X_train[ starting_index : (starting_index + count_real_images) ]

            x_real_images = real_images_raw.reshape( count_real_images, self.W, self.H, self.C )

            y_real_labels = np.ones([count_real_images,1])



            # Grab Generated Images for this training batch

            latent_space_samples = self.sample_latent_space(count_real_images)

            x_generated_images = self.generator.Generator.predict(latent_space_samples)

            y_generated_labels = np.zeros([self.BATCH-count_real_images,1])



            # Combine to train on the discriminator

            x_batch = np.concatenate( [x_real_images, x_generated_images] )

            y_batch = np.concatenate( [y_real_labels, y_generated_labels] )



            # Now, train the discriminator with this batch

            discriminator_loss = self.discriminator.Discriminator.train_on_batch(x_batch,y_batch)[0]

        

            # Generate Noise

            x_latent_space_samples = self.sample_latent_space(self.BATCH)

            y_generated_labels = np.ones([self.BATCH,1])

            generator_loss = self.gan.gan_model.train_on_batch(x_latent_space_samples,y_generated_labels)

            

            if e % self.CHECKPOINT == 0 :

              print ('Epoch: '+str(int(e))+', [Discriminator :: Loss: '+str(discriminator_loss)+'], [ Generator :: Loss: '+str(generator_loss)+']')

              self.plot_checkpoint(e)

                        

        return



    def sample_latent_space(self, instances):

        return np.random.normal(size=(instances,self.LATENT_SPACE_SIZE))



    def plot_checkpoint(self,e):

        noise  = self.sample_latent_space(16)

        images = self.generator.Generator.predict(noise)

        images = (images * 0.5) + 0.5

        images = np.clip(images, 0, 1)

        images = images[:,:,::-1]

        images *= 255

        images = np.clip(images, 0, 255).astype('uint8')        

        f, ax = plt.subplots(4,4, figsize=(16,10))

        for i, img in enumerate(images):

                ax[i//4, i%4].imshow(img)

                ax[i//4, i%4].axis('off')

        plt.show()

                

        return
# Command Line Argument Method

HEIGHT  = 32

WIDTH   = 32

CHANNEL = 3

LATENT_SPACE_SIZE = 100

EPOCHS = 100000

BATCH = 32

CHECKPOINT = 5000

MODEL_TYPE = 3
trainer = Trainer(height=HEIGHT,\

                 width=WIDTH,\

                 channels=CHANNEL,\

                 latent_size=LATENT_SPACE_SIZE,\

                 epochs =EPOCHS,\

                 batch=BATCH,\

                 checkpoint=CHECKPOINT,

                 model_type=MODEL_TYPE)
trainer.train()
images =trainer.X_train[:40]

images = (images * 0.5) + 0.5

images = np.clip(images, 0, 1)

images = images[:,:,::-1]

images *= 255

images = np.clip(images, 0, 255).astype('uint8')   



f, ax = plt.subplots(5,8, figsize=(16,10))

for i, img in enumerate(images):

        ax[i//8, i%8].imshow(img)

        ax[i//8, i%8].axis('off')

        

plt.show()
noise = np.random.normal(0, 1, (40,100))

generated_images = trainer.generator.Generator.predict(noise)

generated_images = (generated_images * 0.5) + 0.5

generated_images = np.clip(generated_images, 0, 1)

generated_images *= 255

generated_images = np.clip(generated_images, 0, 255).astype('uint8')



f, ax = plt.subplots(5,8, figsize=(16,10))

for i, img in enumerate(generated_images):

        ax[i//8, i%8].imshow(img)

        ax[i//8, i%8].axis('off')

        

plt.show()