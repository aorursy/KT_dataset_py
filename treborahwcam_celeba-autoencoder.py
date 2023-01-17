import pickle

import numpy as np

import matplotlib.pyplot as plt
from keras.models import Sequential, Model

from keras.layers import Dense, Dropout, Reshape, Flatten, Input

from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D

import tensorflow as tf
# Encoder

encoder = Sequential();

encoder.add(Conv2D(3, (2, 2), padding = "same", activation = "relu", input_shape = (64, 64, 3,)))

encoder.add(Dropout(0.3))

encoder.add(MaxPooling2D((2, 2)))

encoder.add(Conv2D(5, (2, 2), padding = "same", activation = "relu"))

encoder.add(Dropout(0.3))

encoder.add(MaxPooling2D((2, 2)))

encoder.add(Conv2D(5, (2, 2), padding = "same", activation = "relu"))



encoder.add(Flatten())

encoder.add(Dense(1024, activation="relu"))

encoder.add(Dense(512, activation="relu"))

encoder.add(Dense(256, activation="tanh"))

encoder.add(Reshape((4, 4, 16)))



print(encoder.summary())



# Decoder

decoder = Sequential()

decoder.add(Conv2DTranspose(8, (2, 2), padding = "same", activation = "relu", input_shape = (4, 4, 16)))

decoder.add(UpSampling2D(size = (2, 2)))

decoder.add(Dropout(0.3))

decoder.add(Conv2DTranspose(4, (2, 2), padding = "same", activation = "relu"))

decoder.add(UpSampling2D(size = (2, 2)))

decoder.add(Dropout(0.3))

decoder.add(Conv2DTranspose(4, (2, 2), padding = "same", activation = "relu"))

decoder.add(UpSampling2D(size = (2, 2)))

decoder.add(Dropout(0.3))

decoder.add(Conv2DTranspose(4, (2, 2), padding = "same", activation = "relu"))

decoder.add(Flatten())

decoder.add(Dense(4096, activation="relu"))

decoder.add(Dense(8192, activation="relu"))

decoder.add(Dense(12288, activation="sigmoid"))



decoder.add(Reshape((64, 64, 3)))



print(decoder.summary())



# Create the three models used in training

auto_input = Input((64, 64, 3, ))

x = encoder(auto_input)

auto_output = decoder(x)



autoencoder = Model(auto_input, auto_output)

autoencoder.compile(loss = "mean_squared_error", optimizer= "rmsprop")



print(autoencoder.summary())
# Import the Celeba Dataset



pickle_in = open("/kaggle/input/64x-colour-celeba/64x_Colour.pickle", "rb")

dataset_64x = pickle.load(pickle_in).astype(np.float32)



dataset_64x = np.reshape(dataset_64x, (-1, 64, 64, 3))

for i in range(len(dataset_64x)):

    dataset_64x[i] = (dataset_64x[i].astype(np.float32)) / 255

    

#dataset_64x = dataset_64x / 255



plt.imshow(np.reshape(dataset_64x[0], (64, 64, 3)))
def display_images (autoencoder, dataset, amount, image_size, epoch):

    decoded_imgs = autoencoder.predict(dataset[:amount])

    

    #Show and save a set of reconstructed images from each epoch

    

    plt.figure(figsize=(amount, 4))

    for i in range(amount):

        # Show original image

        ax = plt.subplot(2, amount, i + 1)

        plt.imshow(decoded_imgs[i].reshape(image_size))

        plt.gray()

        ax.get_xaxis().set_visible(False)

        ax.get_yaxis().set_visible(False)

        

    plt.savefig("Epoch#" + (str)(epoch) + ".png")
EPOCHS = 100

BATCH_SIZE = 2048



for e in range(EPOCHS):

    # Train the network

    autoencoder.fit(dataset_64x, dataset_64x, epochs=5, batch_size=BATCH_SIZE, shuffle=True, verbose=1)

    

    # Show a image that was generated

    display_images(autoencoder, dataset_64x, 10, (64, 64, 3), e)