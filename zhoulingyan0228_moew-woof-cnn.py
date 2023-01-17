import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy
import scipy.io.wavfile as wavfile
import sklearn
import sklearn.metrics
import seaborn as sns
import random
import math
import sklearn.utils
import sklearn.metrics
import matplotlib.pyplot as plt
import glob
import os
import scipy
import scipy.signal
import tensorflow as tf
from imblearn.over_sampling import RandomOverSampler
import IPython
WAVE_FOLDER = '../input/cats_dogs'
FRAMERATE = 16000
MAX_WAV_SAMPLES = 20*FRAMERATE
DOWNSAMPLING_SCALE = 1

df = pd.read_csv("../input/train_test_split.csv")
test_cat = df[['test_cat']].dropna().rename(index=str, columns={"test_cat": "file"}).assign(label=0)
test_dog = df[['test_dog']].dropna().rename(index=str, columns={"test_dog": "file"}).assign(label=1)
train_cat = df[['train_cat']].dropna().rename(index=str, columns={"train_cat": "file"}).assign(label=0)
train_dog = df[['train_dog']].dropna().rename(index=str, columns={"train_dog": "file"}).assign(label=1)

test_df = pd.concat([test_cat, test_dog]).reset_index(drop=True)
train_df = pd.concat([train_cat, train_dog]).reset_index(drop=True)
def plot_spectrogram(file):
    x = wavfile.read(file)[1]
    f, t, Sxx = scipy.signal.spectrogram(x)
    plt.pcolormesh(t, f, Sxx)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sample]')


plt.figure(figsize=(50,50))
for i in range(0,10,2):
    plt.subplot(5,2,i+1)
    plot_spectrogram(os.path.join(WAVE_FOLDER, test_cat.iloc[i]['file']))
    plt.subplot(5,2,i+2)
    plot_spectrogram(os.path.join(WAVE_FOLDER, test_dog.iloc[i]['file']))
#wave_raw = wavfile.read(os.path.join(WAVE_FOLDER, test_files[1]))[1]
#wave = np.pad(wave_raw, pad_width=((0, MAX_WAV_SAMPLES-len(wave_raw))), mode='wrap')
#wave = scipy.signal.decimate(wave, 2)
#IPython.display.Audio(wave, rate=FRAMERATE//2)
train_df['label'].plot.hist(bins=2);
random_oversampler = RandomOverSampler()
idx = np.arange(0, len(train_df)).reshape(-1, 1)
idx_sampled, _ = random_oversampler.fit_sample(idx, train_df['label'])
train_files, train_labels = train_df.iloc[idx_sampled.flatten()]['file'].values, train_df.iloc[idx_sampled.flatten()]['label'].values
train_files, train_labels = sklearn.utils.shuffle(train_files, train_labels)
test_files, test_labels = test_df['file'].values, test_df['label'].values
pd.Series(train_labels).plot.hist(bins=2);
def fit_generator(train_files, train_labels, wavs_per_batch=20, augments=5):
    while True:
        maxidx = len(train_files)
        for i in range(0, maxidx, wavs_per_batch):
            waves_batch = []
            labels_batch = []
            for j in range(i, min(maxidx, i+wavs_per_batch)):
                file, label = train_files[j], train_labels[j]
                wave_raw = wavfile.read(os.path.join(WAVE_FOLDER, file))[1]
                wave_raw = wave_raw/np.std(wave_raw)
                length = len(wave_raw)
                waves_batch.append(np.pad(wave_raw, pad_width=((0, MAX_WAV_SAMPLES - length)), mode='wrap'))
                labels_batch.append(label)
                for _ in range(augments):
                    wave_rotated = np.roll(wave_raw, random.randint(0, length))
                    while random.choice([True, False]):
                        wave_rotated += np.roll(wave_raw, random.randint(0, length))
                    wave = np.pad(wave_rotated, pad_width=((0, MAX_WAV_SAMPLES - length)), mode='wrap')
                    #wave = scipy.signal.decimate(wave, DOWNSAMPLING_SCALE)
                    waves_batch.append(wave)
                    labels_batch.append(label)
            yield np.array(waves_batch), np.array(labels_batch)

def validate_generator(test_files, test_labels, wavs_per_batch=20):
    while True:
        maxidx = len(test_files)
        for i in range(0, maxidx, wavs_per_batch):
            waves_batch = []
            labels_batch = []
            for j in range(i, min(maxidx, i+wavs_per_batch)):
                file, label = test_files[j], test_labels[j]
                wave_raw = wavfile.read(os.path.join(WAVE_FOLDER, file))[1]
                wave_raw = wave_raw/np.std(wave_raw)
                length = len(wave_raw)
                left = 0
                right = MAX_WAV_SAMPLES - left - length
                wave = np.pad(wave_raw, pad_width=((left, right)), mode='wrap')
                #wave = scipy.signal.decimate(wave, DOWNSAMPLING_SCALE)
                waves_batch.append(wave)
                labels_batch.append(label)
            yield np.array(waves_batch), np.array(labels_batch)
            
def steps_per_epoch(wavs_per_epoch, wavs_per_batch):
    return int(math.ceil(wavs_per_epoch/wavs_per_batch))
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Reshape((MAX_WAV_SAMPLES//DOWNSAMPLING_SCALE,1), input_shape=(MAX_WAV_SAMPLES//DOWNSAMPLING_SCALE,)))
for i in range(14):
    model.add(tf.keras.layers.Conv1D(32, kernel_size=5, 
                                     padding='same',
                                     activation='relu',
                                     kernel_initializer=tf.keras.initializers.Orthogonal(),
                                    ))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=3, strides=2))
    model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(2, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(0.0005),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy']
             )
WAVS_PER_BATCH = 3
AUGMENTS = 10
EPOCHS=15
model.fit_generator(fit_generator(train_files, train_labels, WAVS_PER_BATCH, AUGMENTS),
                    steps_per_epoch=steps_per_epoch(len(train_files), WAVS_PER_BATCH),
                    epochs = EPOCHS,
                    validation_data=validate_generator(test_files, test_labels, WAVS_PER_BATCH),
                    validation_steps=steps_per_epoch(len(test_files), WAVS_PER_BATCH),
                    verbose=2)
predicted_probs = model.predict_generator(
    validate_generator(test_files, test_labels, WAVS_PER_BATCH),
    steps=steps_per_epoch(len(test_files), WAVS_PER_BATCH))
predicted_classes = np.argmax(predicted_probs, axis=1)
print(sklearn.metrics.accuracy_score(predicted_classes, test_labels))
sns.heatmap(sklearn.metrics.confusion_matrix(predicted_classes, test_labels), annot=True);