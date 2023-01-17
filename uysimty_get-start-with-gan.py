import numpy as np

from tensorflow.keras.datasets.mnist import load_data

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import Conv2D

from tensorflow.keras.layers import Flatten

from tensorflow.keras.layers import Dropout

from tensorflow.keras.layers import LeakyReLU

from tensorflow.keras.layers import Reshape

from tensorflow.keras.layers import Conv2DTranspose

from tensorflow.keras.layers import BatchNormalization

from matplotlib import pyplot

import time

import os

from IPython.display import display, clear_output
def define_discriminator(in_shape=(28,28,1)):

    model = Sequential()

    model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=in_shape))

    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))

    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(1, activation='sigmoid'))

    

    # compile model

    opt = Adam(lr=0.0002, beta_1=0.5)

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model
model = define_discriminator()

model.summary()
def define_generator(input_dim):

    model = Sequential()

    

    # foundation for 7x7 image

    n_nodes = 128 * 7 * 7

    model.add(Dense(n_nodes, input_dim=input_dim))

    model.add(BatchNormalization())

    model.add(LeakyReLU(alpha=0.2))

    model.add(Reshape((7, 7, 128)))

    model.add(Dropout(0.2))

    

    # upsample to 14x14

    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))

    model.add(BatchNormalization())

    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.2))

    

    # upsample to 28x28

    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))

    model.add(BatchNormalization())

    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.2))

    

    model.add(Conv2D(1, (7,7), activation='sigmoid', padding='same'))

    return model
noise_dim = 100

g_model = define_generator(noise_dim)

g_model.summary()
# use for generate random noise to generate random image

def generate_noise(noise_dim, n_samples):

    x_input = np.random.randn(noise_dim * n_samples) # generate random noise 

    x_input = x_input.reshape(n_samples, noise_dim)

    return x_input
def generate_fake_samples(noise_dim, n_samples):

  x_input = generate_noise(noise_dim, n_samples) # generate by random noise

  X = g_model.predict(x_input) # generate image from our model

  y = np.zeros((n_samples, 1)) # mark label to 'fake' as 0

  return X, y
fig = pyplot.figure(figsize=(12, 12))

n_samples = 25

X, _ = generate_fake_samples(100, n_samples)

for i in range(n_samples):

    pyplot.subplot(5, 5, 1 + i)

    pyplot.axis('off')

    pyplot.imshow(X[i, :, :, 0])

    pyplot.draw()
def load_real_samples():

    (trainX, _), (_, _) = load_data() # load mnist dataset

    X = np.expand_dims(trainX, axis=-1) # add gray scale channel to image

    X = X.astype('float32') # convert pixel from ints to floats

    X = X / 255.0 # pixel to between

    return X
def get_real_samples(dataset, idx):

    n_sample = len(idx)

    X = dataset[idx]

    y = np.ones((n_sample, 1)) # mark label to 'real' as 1 

    return X, y
def define_gan(g_model, d_model):

    d_model.trainable = False # don't want to update the decriminator model

  

    # connects discriminator and generator

    model = Sequential()

    model.add(g_model)

    model.add(d_model)

  

    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))

    return model
noise_dim = 100

d_model = define_discriminator()

g_model = define_generator(noise_dim)

gan_model = define_gan(g_model, d_model)

gan_model.summary()
d_history = []
def train_gan(dataset, noise_dim, epochs, batch_size):

    steps = int(dataset.shape[0] / batch_size)

    half_batch = int(batch_size / 2)



    # generate plot slot for real time plot

    fig = pyplot.figure(figsize=(12, 12))

    axs = []

    for i in range(25):

        axs.append(pyplot.subplot(5, 5, 1 + i))



    for epoch in range(epochs):

        for step in range(steps):

            # train our discriminator base from our generator result

            sample_idx = range(step, step+half_batch)

            X_real, y_real = get_real_samples(dataset, sample_idx)

            X_fake, y_fake = generate_fake_samples(noise_dim, half_batch)

            X, y = np.vstack((X_real, X_fake)), np.vstack((y_real, y_fake))

            d_loss, _ = d_model.train_on_batch(X, y)



            # train our GAN to improve our generator

            x_gan = generate_noise(noise_dim, batch_size)

            y_gan = np.ones((batch_size, 1))

            gan_model.train_on_batch(x_gan, y_gan)

        if epoch % 100 == 0: # evaluate every 100 epochs

            # evaluate model test only with 100 output

            evaluate_sample_idx = np.random.randint(0, dataset.shape[0], 50)

            X_real_test, y_real_test = get_real_samples(dataset, evaluate_sample_idx)

            x_fake_test, y_fake_test = generate_fake_samples(noise_dim, 50)

            x_test, y_test = np.vstack((X_real_test, x_fake_test)), np.vstack((y_real_test, y_fake_test))

            _, acc = d_model.evaluate(x_fake_test, y_fake_test, verbose=0)

            d_history.append([acc, epoch])



            fig.suptitle('Discriminal Accuracy: {} at epoch {}'.format(acc, epoch), fontsize=16) # display accuracy and epoch on title



            # plot result in real time

            for i in range(25):

              ax = axs[i]

              ax.cla()

              ax.axis('off')

              ax.imshow(x_fake_test[i, :, :, 0])

            fig.savefig("result_at_epoch_{}.png".format(epoch))

            display(fig)

            clear_output(wait = True) 



  
dataset = load_real_samples()

train_gan(dataset, noise_dim, epochs=1500, batch_size=256)
g_model.save("model.h5")
d_history = np.array(d_history)

pyplot.figure(figsize=(12, 6))

pyplot.plot(d_history[:, 1], d_history[:, 0]) # plot history accuracy

pyplot.show()
x_fake, _ = generate_fake_samples(noise_dim, 25)

fig = pyplot.figure(figsize=(12, 12))

for i in range(25):

    pyplot.subplot(5, 5, 1 + i)

    pyplot.axis('off')

    pyplot.imshow(x_fake[i, :, :, 0])