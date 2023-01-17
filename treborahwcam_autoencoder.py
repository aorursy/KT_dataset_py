#Import livbraries

import numpy as np

import pandas as pd

from keras.datasets import mnist

from keras.models import Model, Sequential

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Flatten, Reshape, Conv2DTranspose

from keras.callbacks import TensorBoard



import matplotlib.pyplot as plt
#Import the data

data = np.load("../input/mnist-numpy/mnist.npz")



#Extract data from npz file

x_train = data["x_train"]

x_test = data["x_test"]



#Normalise data

x_train = x_train.astype("float32") / 255

x_test = x_test.astype("float32") / 255



#Flatten the arrays

x_train = x_train.reshape((len(x_train), 28, 28, 1))

x_test = x_test.reshape((len(x_test), 28, 28, 1))



print(x_train.shape)
encoder = Sequential()

encoder.add(Conv2D(2, (2, 2), input_shape = (28, 28, 1, ), padding="same", activation="relu"))

encoder.add(Dropout(0.3))

encoder.add(MaxPooling2D((2, 2)))

encoder.add(Conv2D(2, (2, 2), padding="same", activation="relu"))

encoder.add(Dropout(0.3))

encoder.add(MaxPooling2D((2, 2)))

encoder.add(Conv2D(2, (2, 2), padding="same", activation="relu"))

encoder.add(Flatten())

encoder.add(Dense(units=128, activation="relu"))

encoder.add(Dense(units=64, activation="sigmoid"))



decoder = Sequential()

decoder.add(Dense(98, input_shape=(64,)))

decoder.add(Reshape((7, 7, 2)))

decoder.add(Conv2DTranspose(2, (2, 2), padding="same", activation="relu"))

decoder.add(UpSampling2D((2, 2)));

decoder.add(Dropout(0.3));

decoder.add(Conv2DTranspose(2, (2, 2), padding="same", activation="relu"))

decoder.add(UpSampling2D((2, 2)));

decoder.add(Dropout(0.3));

decoder.add(Flatten())

decoder.add(Dense(1048, activation="relu"))

decoder.add(Dense(784, activation="sigmoid"))

decoder.add(Reshape((28, 28, 1)))



a_in = Input((28, 28, 1,))

x = encoder(a_in)

a_out = decoder(x)



autoencoder = Model(a_in, a_out)

autoencoder.compile(loss="mean_squared_error", optimizer="rmsprop")



print(autoencoder.summary())
#Train the autoencoder

autoencoder.fit(

    x_train, x_train,

    epochs = 50,

    batch_size = 256,

    shuffle = True,

    validation_data = (x_test, x_test),

    verbose = 2

)
#Show some of the reconstructed images

def show_reconstructed_images(decoder, amount):

    

    rSeed = np.random.random(size=(amount, 64))

    for i in range(amount):

        rSeed[i][0] = i/amount

        

    decoded_imgs = decoder.predict(rSeed)

    

    plt.figure(figsize=(amount, 4))

    for i in range(amount):

        #Show reconstructed image

        ax = plt.subplot(2, amount, i + 1)

        plt.imshow(decoded_imgs[i].reshape(28, 28))

        plt.gray()

        ax.get_xaxis().set_visible(False)

        ax.get_yaxis().set_visible(False)

    

    plt.show()
def show_dense_representation(encoder, amount):

    

    encoded_imgs = encoder.predict(x_test[:amount])

    

    plt.figure(figsize=(amount*2, 2))

    for i in range(amount):

        ax = plt.subplot(1, amount, i + 1)

        plt.imshow(encoded_imgs[i].reshape(4, 16))

        plt.gray()

        ax.get_xaxis().set_visible(False)

        ax.get_yaxis().set_visible(False)

    

    plt.show()
for i in range(2):

    show_reconstructed_images(decoder, 20)

    

show_dense_representation(encoder, 10)