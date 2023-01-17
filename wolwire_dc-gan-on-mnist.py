import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow import keras
from numpy import argmax
from keras.utils import to_categorical
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras import Model
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
import pdb
mnist=tf.keras.datasets.mnist
k=mnist.load_data()
(train_mnist, test) = k
train_mnist_image, train_mnist_label = train_mnist
train_mnist_image = np.array(train_mnist_image)
train_mnist_label = np.array(train_mnist_label)
train_mnist_label = to_categorical(train_mnist_label)
train_mnist_image = np.reshape(train_mnist_image, (-1,28,28,1))
print(train_mnist_label.shape, train_mnist_image.shape)
def gen_image(arr):
    two_d = (np.reshape(arr, (28, 28)))
    plt.imshow(two_d, interpolation='nearest') #cmap=plt.get_cmap('gray')
    return plt
def batch_generator(X, batch_size = 128):
    input_shape = X.shape[0]
    image_list = np.array(range(input_shape))
    np.random.shuffle(image_list)
    iterations = int(input_shape / batch_size)
    image = []
    for i in range(iterations-1):
        batch_list = image_list[i * (batch_size) : (i+1) * (batch_size)]
        image.append(X[batch_list])
    return np.array(image)
def generator_model():
    X_gen_input = Input(shape=(10))
    X = Dense(256, activation='relu')(X_gen_input)
    X = Dropout(.5)(X)
    X = Dense(144, activation='relu')(X)
    X = Reshape((12, 12, 1))(X)
    
    X = Conv2DTranspose(16, (3, 3), strides=(1, 1), padding='valid', activation='relu')(X)
    X = UpSampling2D(size=(2, 2), interpolation='nearest')(X)
    
    X = Conv2DTranspose(8, (3, 3), strides=(1, 1), padding='same', activation='relu')(X)
    X = Dropout(.5)(X)
    
    X = Conv2DTranspose(4, (3, 3), strides=(1, 1), padding='same', activation='relu')(X)
    X = Dropout(.5)(X)
    
    X = Conv2DTranspose(2, (3, 3), strides=(1, 1), padding='same', activation='relu')(X)
    X = Dropout(.5)(X)
    
    output = Conv2DTranspose(1, (3, 3), strides=(1, 1), padding='same', activation='tanh')(X)
    
    model = Model(inputs = X_gen_input, outputs = output)
    return model
def discriminator_model():
    X_dis_input = Input(shape=(28, 28, 1))
    X = Conv2D(2, (3, 3), strides=(1, 1), padding='same', activation='relu')(X_dis_input)
    X = Conv2D(4, (3, 3), strides=(1, 1), padding='same', activation='relu')(X)
    X = Conv2D(8, (3, 3), strides=(1, 1), padding='same', activation='relu')(X)
    X = MaxPooling2D((2, 2), strides=(1, 1))(X)
    X = Conv2D(16, (3, 3), strides=(1, 1), padding='valid', activation='relu')(X)
    X = Flatten()(X)
                                                                             
    X = Dense(256, activation='relu')(X)
    X = Dropout(.5)(X)
    X = Dense(128, activation = 'relu')(X)
    X = Dropout(.5)(X)
    output_dis = Dense(1, activation = 'sigmoid')(X)
    
    model = Model(inputs = X_dis_input, outputs = output_dis)
    return model
def combined_model(generator,discriminator):
    input_noise = Input((10))
    image = generator(input_noise)
    discriminator.trainable = False
    output = discriminator(image)
    model = Model(input_noise, output)
    return model
generator = generator_model()
discriminator = discriminator_model()
discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
combined = combined_model(generator, discriminator)
combined.compile(loss='binary_crossentropy', optimizer='adam')

len_train_set = train_mnist_image.shape[0]

batch_size = 128
EPOCHS = 100

fake_label = np.zeros((batch_size, 1))
real_label = np.ones((batch_size, 1))

for i in range(EPOCHS):
    image_batches = batch_generator((((train_mnist_image)/127.5)-1) , batch_size)
    for real_image_batch in image_batches:
        noise_batch = np.random.normal(0, 1, (batch_size, 10))
        fake_image_batch = generator.predict(noise_batch)
        
        fake_discriminator_loss = discriminator.train_on_batch(fake_image_batch, fake_label)
        real_discriminator_loss = discriminator.train_on_batch(real_image_batch, real_label)
        discriminator_loss  = (fake_discriminator_loss[0] + real_discriminator_loss[0]) * 0.5
        
#         print(noise_batch.shape, noise_batch)
        combined_loss = combined.train_on_batch(noise_batch, real_label)
    
    
    generator.save_weights('./checkpoints/generator_checkpoint')
    discriminator.save_weights('./checkpoints/discriminator_checkpoint')
    combined.save_weights('./checkpoints/combined_checkpoint')
    
    print("total_discriminator_loss: " + str(discriminator_loss))
    print("total_generator_loss: " + str(combined_loss))
    print("epoch: " + str(i))
    evaluation = discriminator.evaluate(generator(np.random.normal(0, 1, (len_train_set, 10))), np.ones((len_train_set, 1)))
images = generator(np.random.normal(0, 1, (batch_size, 10)))*.5 + 0.5
i=0
for image in images: # how many imgs will show from the 3x3 grid
    fig, axs = plt.subplots(nrows=1, sharex=True, figsize=(2, 2))
    two_d = (np.reshape(image, (28, 28)))
    axs.imshow(two_d, interpolation='nearest', cmap=plt.get_cmap('gray'))
    i=i+1
    if i%100 == 0:
        a=input()
        if a=='q':
            break 
