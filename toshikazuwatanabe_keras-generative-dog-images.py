# numpy

import numpy as np

# for operate list

import operator

from functools import reduce

# for models

from keras.models import Sequential, Model

from keras.layers import * # Dense, Conv2D, Flatten, Dropout, LeakyRelu

from keras.optimizers import Adam, SGD

from keras.utils.vis_utils import plot_model

from keras_preprocessing.image import ImageDataGenerator

from keras.models import load_model

# for plot images

import matplotlib.pyplot as plt

# for progress bar

from tqdm import tqdm_notebook as tqdm

# for make zip archive

import shutil

# for mkdir

import pathlib

# for save image to file

from imageio import imsave
PATH_TRAIN_IMAGE = '../input/all-dogs'

SHAPE_IMAGE = (64, 64, 3)

BATCH_SIZE = 32

# size of the latent space

latent_dim = 100

num_epoch = 100

num_batch = 20579 // BATCH_SIZE // 2

print(num_batch)
datagen = ImageDataGenerator(

    horizontal_flip = True,

    rotation_range = 20,

    width_shift_range = 0.2,

    height_shift_range = 0.2,

    preprocessing_function = lambda X: (X-127.5)/127.5,

#     rescale = 1./255

)

data_loader=datagen.flow_from_directory(

    PATH_TRAIN_IMAGE, # directory

    target_size = SHAPE_IMAGE[:-1], 

    class_mode = None,

    batch_size = BATCH_SIZE)
def generate_real_samples(n_samples=BATCH_SIZE):

    return next(data_loader)[:n_samples], np.ones((n_samples, 1))*0.9
def show_images(ary, rows, cols):

    plt.figure(figsize=(cols*3, rows*3))

    for row in range(rows):

        for col in range(cols):

            plt.subplot(rows, cols, row*cols+col+1)

            img = (ary[row*cols+col, :] + 1) / 2

#             img = ary[row*cols+col]

            plt.axis('off')

            plt.title(f'{row*cols+col}')

            plt.imshow(img)

    plt.show()
data, y = generate_real_samples()

print('shape of data:', data.shape) # => (32, 64, 64, 3)

print('min, max of data:', data.min(), data.max()) # => 0.0 1.0

print('shape of y', y.shape) # => (32, 1)

print('min, max of y', y.min(), y.max()) # => 1.0 1.0

print('head 5 of y', y[:5]) # => [[1.] [1.] ...]



show_images(data,2, 5)
def define_discriminator():

    model = Sequential([

        InputLayer(input_shape=SHAPE_IMAGE),

        Conv2D(32, kernel_size=3, strides=2, padding='same'),

        LeakyReLU(alpha=0.2),

        Dropout(0.25),

        Conv2D(64, kernel_size=3, strides=2, padding='same'),

        ZeroPadding2D(padding=((0,1),(0,1))),

        LeakyReLU(alpha=0.2),

        BatchNormalization(momentum=0.8),

        Dropout(0.25),

        Conv2D(128, kernel_size=3, strides=2, padding='same'),

        LeakyReLU(alpha=0.2),

        Dropout(0.25),

        BatchNormalization(momentum=0.8),

        Conv2D(256, kernel_size=3, strides=2, padding='same'),

        LeakyReLU(alpha=0.2),

        Dropout(0.25),

        Conv2D(512, kernel_size=3, strides=2, padding='same'),

        LeakyReLU(alpha=0.2),

        Dropout(0.25),

        Flatten(),

        Dense(1, activation='sigmoid')

    ])

    

    return model
# compile discriminator

discriminator = define_discriminator()

discriminator_opt = Adam(lr=0.0002, beta_1=0.5)

discriminator.compile(loss='binary_crossentropy', optimizer=discriminator_opt)

discriminator.summary()
def define_generator():

    struct_ = (64, 8, 8)

    n_nodes = reduce(operator.mul, struct_) # (reduce '* struct_)

    print(f'generator input dim={n_nodes}')

    model = Sequential([

        Dense(n_nodes, activation='relu', input_shape=(latent_dim,)),

        Reshape((*struct_[1:], struct_[0])),

        BatchNormalization(momentum=0.8),

        

        # upsample to 16x16

        UpSampling2D(),

        Conv2D(struct_[0], kernel_size=3, padding='same'),

        Activation('relu'),

        BatchNormalization(momentum=0.8),

        # upsample to 32x32

        UpSampling2D(),

        Conv2D(struct_[0]//2, kernel_size=3, padding='same'),

        Activation('relu'),

        BatchNormalization(momentum=0.8),

        # upsample to 64x64

        UpSampling2D(),

        Conv2D(struct_[0]//4, kernel_size=3, padding='same'),

        Activation('relu'),

        BatchNormalization(momentum=0.8),

        Conv2D(3, kernel_size=3, padding='same'),

        Activation('tanh'),

    ])

    

    return model
# make generator

generator = define_generator()

generator.summary()
# generate points in latent space as input for the generator

def generate_latent_points(latent_dim, n_samples):

#     noize = np.random.normal(0.5, 1, (n_samples, latent_dim))

    noize = np.random.uniform(-1, 1, (n_samples, latent_dim))

    return noize
# use the generator to generate n fake examples, with class labels

def generate_fake_samples(g_model, latent_dim, n_samples):

    # generate points in latent space

    x_input = generate_latent_points(latent_dim, n_samples)

    # predict outputs

    X = g_model.predict(x_input)

    # create 'fake' class labels (0)

    y = np.zeros((n_samples, 1))

    return X, y
# generator output test

X, y = generate_fake_samples(generator, latent_dim, 10)

show_images(X, 2, 5)
# define the combined generator and discriminator model, for updating the generator

def define_gan(g_model, d_model):

    d_model_fixed = Model(inputs=d_model.inputs, outputs=d_model.outputs)

    d_model_fixed.trainable = False

    # connect them

    model = Sequential([

        InputLayer(input_shape=(latent_dim,)),

        g_model,

        d_model_fixed

    ])

    # compile model

    opt = Adam(lr=0.0002, beta_1=0.5)

    model.compile(loss='binary_crossentropy', optimizer=opt)

    return model
# compile gan

gan = define_gan(generator, discriminator)

gan.summary()
def train_discriminator():

    # get randomly selected 'real' samples

    X_real, y_real = generate_real_samples(BATCH_SIZE//2)

    loss_real = discriminator.train_on_batch(X_real, y_real)

    # generate 'fake' examples

    X_fake, y_fake = generate_fake_samples(generator, latent_dim, BATCH_SIZE//2)

    loss_fake = discriminator.train_on_batch(X_fake, y_fake)

    return (loss_real+loss_fake)*0.5

# train_discriminator()
def train_gan(num_loop=1):

    # prepare points in latent space as input for the generator

    X = generate_latent_points(latent_dim, BATCH_SIZE)

    # create inverted labels for the fake samples

    y = np.ones((BATCH_SIZE, 1))*0.9

    for i in range(num_loop):

        loss = gan.train_on_batch(X, y)

    return loss

# train_gan()
# train all

history = np.zeros((num_epoch, num_batch, 2))

dogs_at_epoch = np.zeros((num_epoch, *SHAPE_IMAGE))



for i in tqdm(range(num_epoch), desc='epoch'):

    data_loader.reset()

    pbar_batch = tqdm(range(num_batch), desc='batch')

    

    for j in pbar_batch:

        d_loss = train_discriminator()

        g_loss = train_gan()

        pbar_batch.set_description(f'{i:>2}, d_loss:{d_loss:.2}, g_loss:{g_loss:.2}')

        history[i, j, :] = d_loss, g_loss

        

    generated_imgs = generate_fake_samples(generator, latent_dim, 5)[0]

    show_images(generated_imgs, 1, 5)

    dogs_at_epoch[i, :] = generated_imgs[0,:]
# show_images(dogs_at_epoch[:, :, :, :], num_epoch//5, 5)

plt.plot(history[:,-1,:])
# generate images

latent_points = generate_latent_points(latent_dim, 10000)

# generate images

X = generator.predict(latent_points)



print(X.shape, X[0].min(), X[0].max())
show_images(X, 2, 5)
imgs = [((img+1) * 127.5).astype(np.uint8) for img in X]

np.array(imgs).min(), np.array(imgs).max()
IMG_DIR = pathlib.Path('images')

if not IMG_DIR.exists():

    IMG_DIR.mkdir()



for n in range(len(imgs)):

    imsave(IMG_DIR/f'dog_{n}.png', imgs[n])
shutil.make_archive('images', 'zip', 'images')
!rm -rf images
!ls