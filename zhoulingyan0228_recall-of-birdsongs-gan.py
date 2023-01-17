import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import IPython
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import sklearn
import sklearn.utils
import scipy
import scipy.signal
import scipy.io
import scipy.io.wavfile as wavfile
import glob
import itertools
import sys
tf.logging.set_verbosity(tf.logging.WARN)
files = glob.glob("../input/Training/*.wav")
files.sort()
FRAMERATE = 16000
LENGTH = 16000
PADDED_LENGTH = 2**14
for f in files:
    IPython.display.display(IPython.display.Audio(f))
plt.figure(figsize=(30,30))
for i, f in enumerate(files):
    _, signal= wavfile.read(f)
    plt.subplot(5, 5, i+1)
    signal = signal[:LENGTH]
    plt.specgram(signal, NFFT=256, Fs=2, Fc=0, noverlap=128)
files.pop(0)
bird_songs = np.concatenate([wavfile.read(f)[1][:FRAMERATE] for f in files])
bird_songs = bird_songs / np.max(bird_songs)
SEED_LENGTH = 4096

encoder = tf.keras.models.Sequential()
encoder.add(tf.keras.layers.InputLayer(input_shape=(LENGTH,)))
encoder.add(tf.keras.layers.Reshape((LENGTH, 1)))
encoder.add(tf.keras.layers.ZeroPadding1D((0,PADDED_LENGTH-LENGTH)))
for s, k, n in ([4, 25, 16],[4, 25, 32],[4, 15, 64]):
    encoder.add(tf.keras.layers.Conv1D(n, kernel_size=k, strides=s, padding='same'))
    encoder.add(tf.keras.layers.LeakyReLU())
    encoder.add(tf.keras.layers.BatchNormalization())
encoder.add(tf.keras.layers.Reshape((256, 64, 1)))
for s, k, n in ([(4,2), (15,5), 16],[(4,2), (15,5), 16]):
    encoder.add(tf.keras.layers.Conv2D(n, kernel_size=k, strides=s, padding='same'))
    encoder.add(tf.keras.layers.LeakyReLU())
    encoder.add(tf.keras.layers.BatchNormalization())
encoder.add(tf.keras.layers.Flatten())
encoder.summary()

decoder = tf.keras.models.Sequential()
decoder.add(tf.keras.layers.InputLayer(input_shape=(SEED_LENGTH,)))
decoder.add(tf.keras.layers.Reshape((16,16,16)))
for s, k, n in reversed(([(4,2), (15,5), 16],[(4,2), (15,5), 16])):
    decoder.add(tf.keras.layers.Conv2DTranspose(n, kernel_size=k, strides=s, padding='same'))
    decoder.add(tf.keras.layers.LeakyReLU())
    decoder.add(tf.keras.layers.BatchNormalization())
decoder.add(tf.keras.layers.Conv2DTranspose(1, kernel_size=1, strides=1, padding='same'))
decoder.add(tf.keras.layers.Reshape((256, 1, 64)))
for s, k, n in reversed(([4, 25, 16],[4, 25, 32],[4, 15, 64])):
    decoder.add(tf.keras.layers.Conv2DTranspose(n, kernel_size=(k,1), strides=(s,1), padding='same'))
    decoder.add(tf.keras.layers.LeakyReLU())
    decoder.add(tf.keras.layers.BatchNormalization())
decoder.add(tf.keras.layers.Conv2DTranspose(1, kernel_size=(1,1), strides=(1,1), padding='same'))
decoder.add(tf.keras.layers.Reshape((PADDED_LENGTH, 1)))
decoder.add(tf.keras.layers.Cropping1D(((0, PADDED_LENGTH-LENGTH))))
decoder.add(tf.keras.layers.Reshape((LENGTH, )))
decoder.add(tf.keras.layers.Activation('tanh'))
decoder.summary()

cae = tf.keras.models.Sequential()
cae.add(tf.keras.layers.InputLayer(input_shape=(LENGTH,)))
cae.add(encoder)
cae.add(decoder)
cae.summary()
cae.compile(loss=tf.keras.losses.mean_squared_error,
            optimizer=tf.keras.optimizers.Adam(0.001)
           )
def cae_fit_generator(bird_songs, batch_size=32):
    while True:
        recorded = [bird_songs[x:x+LENGTH] for x in np.random.randint(0,len(bird_songs)-LENGTH, size=batch_size)]
        recorded = np.array(recorded)
        yield recorded, recorded

cae.fit_generator(cae_fit_generator(bird_songs),
                  epochs=10, 
                  steps_per_epoch=2000,
                  callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss',
                              min_delta=0,
                              patience=0,
                              verbose=0, mode='auto')
                      ],
                  verbose=2)
def set_trainable(model, trainable=True):
    model.trainable = trainable
    
def make_noisy_labels(size, label=0, flipperc = 0.1):
    flip = int(size*flipperc)
    #return #np.random.uniform(-0.3, 0.3, size) + np.array([label]*(size-flip)+[1-label]*(flip))
    return np.array([label]*(size-flip)+[1-label]*(flip))

generator_base = tf.keras.models.Sequential()
generator_base.add(tf.keras.layers.InputLayer(input_shape=(SEED_LENGTH,)))
generator_base.add(tf.keras.layers.Dropout(0.2))
generator_base.add(tf.keras.layers.Dense(SEED_LENGTH, activation='tanh'))
generator_base.summary()

generator = tf.keras.models.Sequential()
generator.add(tf.keras.layers.InputLayer(input_shape=(SEED_LENGTH,)))
generator.add(generator_base)
generator.add(decoder)
set_trainable(decoder, False)
generator.summary()

discriminator_top = tf.keras.models.Sequential()
discriminator_top.add(tf.keras.layers.InputLayer(input_shape=(SEED_LENGTH,)))
discriminator_top.add(tf.keras.layers.Dense(1024))
discriminator_top.add(tf.keras.layers.LeakyReLU())
discriminator_top.add(tf.keras.layers.Dense(1, activation='sigmoid'))
discriminator_top.summary()

discriminator = tf.keras.models.Sequential()
discriminator.add(tf.keras.layers.InputLayer(input_shape=(LENGTH,)))
discriminator.add(encoder)
discriminator.add(discriminator_top)
set_trainable(encoder, False)
#set_trainable(discriminator_top, True)
discriminator.summary()
discriminator.compile(loss='mse',
            optimizer=tf.keras.optimizers.Adam(0.00001),
            metrics=[]
           )

gan = tf.keras.models.Sequential()
gan.add(tf.keras.layers.InputLayer(input_shape=(SEED_LENGTH,)))
gan.add(generator)
gan.add(discriminator)
set_trainable(encoder, False)
set_trainable(decoder, False)
set_trainable(generator_base, True)
set_trainable(discriminator_top, False)
gan.summary()
gan.compile(loss='mse',
            optimizer=tf.keras.optimizers.Adam(0.001),
            metrics=[]
           )
N_EACHCLASS=20
d_losses = []
g_losses = []
PRINT_INTERVAL=2000
N_D_TRAIN = 5
N_G_TRAIN = 2
for i in range(20*PRINT_INTERVAL+1):
    seeds = np.random.normal(0, 0.5, (N_EACHCLASS, SEED_LENGTH))
    generated = generator.predict(seeds)
    recorded = np.stack([bird_songs[x: x+LENGTH] for x in np.random.randint(0,len(bird_songs)-LENGTH,(N_EACHCLASS))])
    d_loss = discriminator.test_on_batch(np.concatenate((generated, recorded)), np.concatenate((make_noisy_labels(N_EACHCLASS, 0), make_noisy_labels(N_EACHCLASS, 1,0))))
    g_loss = gan.test_on_batch(seeds, make_noisy_labels(N_EACHCLASS, 1))
    
    j=N_D_TRAIN
    set_trainable(encoder, False)
    set_trainable(discriminator_top, True)
    #set_trainable(discriminator, True)
    while d_loss>0.1 and j>0:
        j -= 1
        seeds = np.random.normal(0, 0.5, (N_EACHCLASS, SEED_LENGTH))
        generated = generator.predict(seeds)
        recorded = np.stack([bird_songs[x: x+LENGTH] for x in np.random.randint(0,len(bird_songs)-LENGTH,(N_EACHCLASS))])
        discriminator.train_on_batch(np.concatenate((generated, recorded)), np.concatenate((make_noisy_labels(N_EACHCLASS, 0), make_noisy_labels(N_EACHCLASS, 1,0))))
        
        seeds = np.random.normal(0, 0.5, (N_EACHCLASS, SEED_LENGTH))
        generated = generator.predict(seeds)
        recorded = np.stack([bird_songs[x: x+LENGTH] for x in np.random.randint(0,len(bird_songs)-LENGTH,(N_EACHCLASS))])
        d_loss = discriminator.test_on_batch(np.concatenate((generated, recorded)), np.concatenate((make_noisy_labels(N_EACHCLASS, 0), make_noisy_labels(N_EACHCLASS, 1,0))))

    j=N_G_TRAIN
    set_trainable(encoder, False)
    set_trainable(decoder, False)
    set_trainable(discriminator_top, False)
    set_trainable(generator_base, True)
    while g_loss>0.5 and j>0:
        j -= 1
        seeds = np.random.normal(0, 0.5, (N_EACHCLASS, SEED_LENGTH))
        gan.train_on_batch(seeds, make_noisy_labels(N_EACHCLASS, 1))
        seeds = np.random.normal(0, 0.5, (N_EACHCLASS, SEED_LENGTH))
        g_loss = gan.test_on_batch(seeds, make_noisy_labels(N_EACHCLASS, 1))
    d_losses.append(d_loss)
    g_losses.append(g_loss)
    if i%PRINT_INTERVAL == PRINT_INTERVAL-1:
        #print('discriminator loss=', d_loss)
        #print('gan loss=', g_loss)
        #print('discriminator weights=', discriminator_top.get_weights())
        plt.plot(np.arange(i+1-PRINT_INTERVAL, i+1), d_losses, 'r--')
        plt.plot(np.arange(i+1-PRINT_INTERVAL, i+1), g_losses, 'b-')
        plt.show()
        d_losses = []
        g_losses = []
seeds = np.random.uniform(-1, 1, (5, SEED_LENGTH))
generated = generator.predict_on_batch(seeds)
plt.figure(figsize=(20,5))
for i, signal in enumerate(generated):
    IPython.display.display(IPython.display.Audio(signal.flatten(), rate=FRAMERATE))
    plt.subplot(2,5,i+1)
    plt.plot(signal)
    plt.subplot(2,5,5+i+1)
    plt.specgram(signal, NFFT=256, Fs=2, Fc=0, noverlap=128)