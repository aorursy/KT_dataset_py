import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import os

import time

import tensorflow as tf



import keras

from keras.models import Sequential, Model

from keras.layers import Conv2D, Conv2DTranspose, Dropout, Dense, Flatten, BatchNormalization, MaxPooling2D, AvgPool2D

from keras.layers import Input, Lambda, UpSampling2D, concatenate, Activation, Embedding, Reshape, Concatenate, multiply

from keras.optimizers import RMSprop, SGD, Adam

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from keras.layers.advanced_activations import LeakyReLU

from keras.callbacks import CSVLogger, ModelCheckpoint

from keras.utils.np_utils import to_categorical



from sklearn.model_selection import train_test_split
# Load the data

Dig_MNIST = pd.read_csv("../input/Kannada-MNIST/Dig-MNIST.csv")

sample_submission = pd.read_csv("../input/Kannada-MNIST/sample_submission.csv")

test = pd.read_csv("../input/Kannada-MNIST/test.csv")

train = pd.read_csv("../input/Kannada-MNIST/train.csv")



x_train = train.drop('label', axis=1).to_numpy().reshape((60000, 28, 28))

y_train = train['label'].to_numpy()



img_shape = (28, 28, 1)

z_dim = 100

num_classes = 10
def build_generator(z_dim):

    

    model = Sequential()

    model.add(Dense(7*7*256, input_shape=(z_dim, )))

    model.add(Reshape((7, 7, 256)))

    model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'))

    model.add(BatchNormalization())

    model.add(LeakyReLU(alpha=0.01))

    model.add(Conv2DTranspose(64, kernel_size=3, strides=1, padding='same'))

    model.add(BatchNormalization())

    model.add(LeakyReLU(alpha=0.01))

    model.add(Conv2DTranspose(1, kernel_size=3, strides=2, padding='same'))

    model.add(Activation('tanh'))

    

    z = Input(shape=(z_dim, ))



    label = Input(shape=(1,), dtype='int32')

    

    label_embedding = Embedding(num_classes, z_dim, input_length=1)(label)

    

    label_embedding = Flatten()(label_embedding)



    joined_representation = multiply([z, label_embedding])

    

    img = model(joined_representation)

    

    return Model([z, label], img)
def build_discriminator(img_shape):

    

    model = Sequential()

    

    model.add(Conv2D(64, kernel_size=3, strides=2, padding='same', input_shape=(28, 28, 2)))

    model.add(LeakyReLU(alpha=0.01))

    model.add(Conv2D(64, kernel_size=3, strides=2, padding='same'))

    model.add(BatchNormalization())

    model.add(LeakyReLU(alpha=0.01))

    model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))

    model.add(BatchNormalization())

    model.add(LeakyReLU(alpha=0.01))

    model.add(Flatten())

    model.add(Dense(1, activation='sigmoid'))

    

    img = Input(shape=img_shape)



    label = Input(shape=(1,), dtype='int32')

    

    label_embedding = Embedding(input_dim=num_classes, output_dim=np.prod(img_shape), input_length=1)(label)

    

    label_embedding = Flatten()(label_embedding)

    

    label_embedding = Reshape(img_shape)(label_embedding)

    

    concatenated = Concatenate(axis=-1)([img, label_embedding])

    

    prediction = model(concatenated)

    

    return Model([img, label], prediction)
disc = build_discriminator(img_shape)

disc.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=Adam())



gen = build_generator(z_dim)

z = Input(shape=(z_dim,))

label = Input(shape=(1,))



img = gen([z, label])



disc.trainable = False



prediction = disc([img, label])



cgan = Model([z, label], prediction)

cgan.compile(loss='binary_crossentropy', optimizer=Adam())
def sample_images(image_grid_rows=2, image_grid_columns=5):

    z = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, z_dim))

    labels = np.arange(0, 10).reshape(-1, 1)

    gen_imgs = gen.predict([z, labels])

    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(image_grid_rows, image_grid_columns, figsize=(10,4), sharey=True, sharex=True)

    cnt = 0

    for i in range(image_grid_rows):

        for j in range(image_grid_columns):

            axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')

            axs[i,j].axis('off')

            axs[i,j].set_title("Digit: %d" % labels[cnt])

            cnt += 1
accuracies = []

losses = []



def train(iterations, batch_size, sample_interval):

    

    X_train = (x_train - 127.5) / 127.5

    X_train = np.expand_dims(X_train, axis=3)

    

    real = np.ones(shape=(batch_size, 1))

    fake = np.zeros(shape=(batch_size, 1))

    

    for iteration in range(iterations):

        

        idx = np.random.randint(0, X_train.shape[0], batch_size)

        imgs, labels = X_train[idx], y_train[idx]

        

        z = np.random.normal(0, 1, size=(batch_size, z_dim))

        gen_imgs = gen.predict([z, labels])

        

        d_loss_real = disc.train_on_batch([imgs, labels], real)

        d_loss_fake = disc.train_on_batch([gen_imgs, labels], fake)

        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        

        z = np.random.normal(0, 1, size=(batch_size, z_dim))

        labels = np.random.randint(0, num_classes, batch_size).reshape(-1, 1)

        

        g_loss = cgan.train_on_batch([z, labels], real)

        

        if iteration % sample_interval == 0:

            print('{} [D loss: {}, accuracy: {:.2f}] [G loss: {}]'.format(iteration, d_loss[0], 100 * d_loss[1], g_loss))

        

            losses.append((d_loss[0], g_loss))

            accuracies.append(d_loss[1])

            

            sample_images()

    
iterations = 40000

batch_size = 64

sample_interval = 1000



train(iterations, batch_size, sample_interval)