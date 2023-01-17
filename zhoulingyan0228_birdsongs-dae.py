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
files = glob.glob("../input/Training/*.wav")
files.sort()
files.pop(0)
FRAMERATE = 16000
LENGTH = 16000
PADDED_LENGTH = 2**14
bird_songs = np.concatenate([wavfile.read(f)[1] for f in files])
bird_songs = bird_songs / np.max(bird_songs)
ENCODED_FEATURES = 4096

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
decoder.add(tf.keras.layers.InputLayer(input_shape=(ENCODED_FEATURES,)))
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
def fit_generator(bird_songs, batch_size=64):
    while True:
        recorded = np.array([bird_songs[x: x+LENGTH] for x in np.random.randint(0,len(bird_songs)-LENGTH,(batch_size))])
        noised = np.array([np.random.normal(0, 0.1, LENGTH)+seg for seg in recorded])
        yield noised, recorded
BATCH_SIZE = 32
STEPS_PER_EPOCH = 1000
EPOCHS = 50
cae.fit_generator(fit_generator(bird_songs),
                  epochs=EPOCHS, 
                  steps_per_epoch=STEPS_PER_EPOCH,
                  callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss',
                              min_delta=0,
                              patience=0,
                              verbose=0, mode='auto')
                      ],
                  verbose=2)
SAMPLE = 5
segments = np.stack([np.random.normal(0, 0.1, LENGTH)+bird_songs[x: x+LENGTH] for x in np.random.randint(0,len(bird_songs)-LENGTH,(SAMPLE))])
reproduced = cae.predict(segments)

plt.figure(figsize=(20,5))
for i, s, r in zip(np.arange(5), segments, reproduced):
    s = s.flatten()
    r = r.flatten()
    plt.subplot(SAMPLE,4,i*4+1)
    plt.plot(s)
    plt.subplot(SAMPLE,4,i*4+2)
    plt.specgram(s, NFFT=256, Fs=2, Fc=0, noverlap=128)
    plt.subplot(SAMPLE,4,i*4+3)
    plt.plot(r)
    plt.subplot(SAMPLE,4,i*4+4)
    plt.specgram(r, NFFT=256, Fs=2, Fc=0, noverlap=128)

for i, signal in enumerate(reproduced):
    IPython.display.display(IPython.display.Audio(signal.flatten(), rate=FRAMERATE))
segments = np.stack([np.random.normal(0, 0.1, LENGTH)+bird_songs[x: x+LENGTH] for x in np.random.randint(0,len(bird_songs)-LENGTH,(2))])
reproduced1,reproduced2 = encoder.predict_on_batch(segments)
steps = 10

delta = (reproduced2 - reproduced1) / steps
interpolated = np.array([reproduced1+delta*i for i in range(steps+1)])
generated = decoder.predict_on_batch(interpolated)

plt.figure(figsize=(20,5))
for i, signal in enumerate(generated):
    signal =  signal.flatten()
    plt.subplot(2,steps+1, i+1)
    plt.plot(signal)
    plt.subplot(2,steps+1, i+(steps+1)+1)
    plt.specgram(signal, NFFT=256, Fs=2, Fc=0, noverlap=128)