import tensorflow as tf

import matplotlib.pyplot as plt

import numpy as np

import glob

import os

%matplotlib inline
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5
BATCH_SIZE = 256

BUFFER_SIZE = 60000
datasets = tf.data.Dataset.from_tensor_slices(train_images)
datasets = datasets.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
# 创建生成器

def generator_model():

    model = tf.keras.Sequential()

    # 生成器一般不用偏执 use_bias False

    model.add(tf.keras.layers.Dense(7*7*256, input_shape=(100,), use_bias=False))

    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Reshape((7, 7, 256)))

    

    model.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding="same", use_bias=False))

    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.LeakyReLU())

    

    model.add(tf.keras.layers.Conv2DTranspose(64, (5, 5 ), strides=(2, 2), padding="same", use_bias=False))

    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.LeakyReLU())

    

    model.add(tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding="same", use_bias=False))

    model.add(tf.keras.layers.Activation(tf.nn.tanh))

   

    return model
# 创建辨别器

def discriminator_model():

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same", input_shape=(28, 28, 1)))

    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Dropout(0.3))

    

    model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same"))

    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Dropout(0.3))

    

    model.add(tf.keras.layers.Conv2D(256, (5, 5), strides=(2, 2), padding="same"))

    model.add(tf.keras.layers.LeakyReLU())

    

    model.add(tf.keras.layers.Flatten())

    

    model.add(tf.keras.layers.Dense(1))

    

    return model
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def discriminator_loss(real_out, fake_out):

    real_loss = cross_entropy(tf.ones_like(real_out), real_out)

    fake_loss = cross_entropy(tf.zeros_like(fake_out), fake_out)

    return real_loss + fake_loss
def generator_loss(fake_out):

    return cross_entropy(tf.ones_like(fake_out), fake_out)
generator_opt = tf.keras.optimizers.Adam(1e-4) 

discriminator_opt = tf.keras.optimizers.Adam(1e-4) 
EPOCHS = 100

noise_dim = 100

num_exp_to_generator =  16

seed = tf.random.normal([num_exp_to_generator, noise_dim])
generator = generator_model()

discriminator = discriminator_model()
def train_step(images):

    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        real_out = discriminator(images)

        gen_images = generator(noise)

        fake_out = discriminator(gen_images)

        gen_loss = generator_loss(fake_out)

        disc_loss = discriminator_loss(real_out, fake_out)

    gradient_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)

    gradient_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_opt.apply_gradients(zip(gradient_gen, generator.trainable_variables))

    discriminator_opt.apply_gradients(zip(gradient_disc, discriminator.trainable_variables))
def generator_plot_images(model, test_noise):

    pre_images = model(test_noise)

    fig = plt.figure(figsize=(4, 4))

    for i in range(pre_images.shape[0]):

        plt.subplot(4, 4, i+1)

        plt.imshow((pre_images[i, :, :, 0] + 1)/2, cmap="gray")

        plt.axis("off")

    plt.show()
def train(dataset, epochs):

    for epoch in range(epochs):

        for image_batch in dataset:

            train_step(image_batch)

            print('.', end="")

        generator_plot_images(generator, seed)
train(datasets, EPOCHS)