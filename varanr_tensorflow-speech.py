# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from sklearn.utils import shuffle
import os
#print(os.listdir("../input"))
import random
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import IPython.display as ipd
from tqdm import tqdm
%matplotlib inline
# Any results you write to the current directory are saved as output.
print(os.listdir("../input/train"))
loc = "../input/train/audio/"
def load_files(path):
    train_labels = os.listdir(path)
    train_labels.remove('_background_noise_')

    labels_to_keep = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence']

    train_file_labels = dict()
    for label in train_labels:
        files = os.listdir(path + '/' + label)
        for f in files:
            train_file_labels[label + '/' + f] = label

    train = pd.DataFrame.from_dict(train_file_labels, orient='index')
    train = train.reset_index(drop=False)
    train = train.rename(columns={'index': 'file', 0: 'folder'})
    train = train[['folder', 'file']]
    train = train.sort_values('file')
    train = train.reset_index(drop=True)
    
    def remove_label_from_file(label, fname):
        return path + label + '/' + fname[len(label)+1:]

    train['file'] = train.apply(lambda x: remove_label_from_file(*x), axis=1)
    train['label'] = train['folder'].apply(lambda x: x if x in labels_to_keep else 'unknown')

    labels_to_keep.append('unknown')

    return train, labels_to_keep
train, labels_keep = load_files(loc)
word2id = dict((c,i) for i,c in enumerate(sorted(labels_keep)))
unk_files = train.loc[train['label'] == 'unknown']['file'].values
unk_files = random.sample(list(unk_files), 1000)

def extract_feature(path):
    X, sample_rate = librosa.load(path)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz

def parse_audio_files(files, word2id, unk = False):
    # n: number of classes
    features = np.empty((0,193))
    one_hot = np.zeros(shape = (len(files), word2id[max(word2id)]))
    print(one_hot.shape)
    for i in tqdm(range(len(files))):
        f = files[i]
        mfccs, chroma, mel, contrast,tonnetz = extract_feature(f)
        ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
        features = np.vstack([features,ext_features])
        if unk == True:
            l = word2id['unknown']
            one_hot[i][l] = 1.
        else:
            l = word2id[f.split('/')[-2]]
            one_hot[i][l] = 1.
    return np.array(features), one_hot
files = train.loc[train['label'] != 'unknown']['file'].values
print(len(files))
print(files[:10])
train_audio_path = '../input/train/audio/'
filename = '/tree/24ed94ab_nohash_0.wav' 
sample_rate, audio = wavfile.read(str(train_audio_path) + filename)
plt.figure(figsize = (15, 4))
plt.plot(audio)
ipd.Audio(audio, rate=sample_rate)
audio_chunks = []
n_chunks = int(audio.shape[0]/320)
for i in range(n_chunks):
    chunk = audio[i*320: (i+1)*320]
    audio_chunks.append(chunk)
audio_chunk = np.array(audio_chunks)
def log_specgram(audio, sample_rate, window_size=10,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    _, _, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return np.log(spec.T.astype(np.float32) + eps)
spectrogram = log_specgram(audio, sample_rate, 10, 0)
spec = spectrogram.T
print(spec.shape)
plt.figure(figsize = (15,4))
plt.imshow(spec, aspect='auto', origin='lower')
labels = sorted(labels_keep)
word2id = dict((c,i) for i,c in enumerate(labels))
label = train['label'].values
label = [word2id[l] for l in label]
print(labels)
def make_one_hot(seq, n):
    # n --> vocab size
    seq_new = np.zeros(shape = (len(seq), n))
    for i,s in enumerate(seq):
        seq_new[i][s] = 1.
    return seq_new
one_hot_l = make_one_hot(label, 12)
paths = []
folders = train['folder']
files = train['file']
for i in range(len(files)):
    path = str(files[i])
    paths.append(path)
def audio_to_data(path):
    # we take a single path and convert it into data
    sample_rate, audio = wavfile.read(path)
    spectrogram = log_specgram(audio, sample_rate, 10, 0)
    return spectrogram.T

def paths_to_data(paths,labels):
    data = np.zeros(shape = (len(paths), 81, 100))
    indexes = []
    for i in tqdm(range(len(paths))):
        audio = audio_to_data(paths[i])
        if audio.shape != (81,100):
            indexes.append(i)
        else:
            data[i] = audio
    final_labels = [l for i,l in enumerate(labels) if i not in indexes]
    print('Number of instances with inconsistent shape:', len(indexes))
    return data[:len(data)-len(indexes)], final_labels, indexes
d,l,indexes = paths_to_data(paths,one_hot_l)
labels = np.zeros(shape = [d.shape[0], len(l[0])])
for i,array in enumerate(l):
    for j, element in enumerate(array):
        labels[i][j] = element
print(d[0].shape)
print(labels[0].shape)
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Activation, Input, Add
#model 1
model = Sequential()
model.add(LSTM(256, input_shape = (81, 100)))
model.add(Dense(1028))
#model.add(Dropout(0.2))
model.add(Dense(128))
#model.add(Dropout(0.2))
model.add(Dense(12, activation = 'softmax'))
model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.summary()
model.fit(d, labels, batch_size=1024, epochs=10)
#model 2
model_2 = Sequential()

drpt_val=0.75

model_2.add(LSTM(256, input_shape = (81, 100)))
model_2.add(Dropout(0.25))
model_2.add(Activation('tanh'))

model_2.add(Dense(1024))
model_2.add(BatchNormalization())
model_2.add(Dropout(drpt_val))

#model_2.add(Dense(1024))
#model_2.add(BatchNormalization())
#model_2.add(Dropout(0.25))

model_2.add(Dense(512))
model_2.add(BatchNormalization())
model_2.add(Dropout(drpt_val))

model_2.add(Dense(256))
model_2.add(BatchNormalization())
model_2.add(Dropout(drpt_val))

model_2.add(Dense(128))
model_2.add(BatchNormalization())
model_2.add(Dropout(drpt_val))

model_2.add(Dense(12,activation='softmax'))
model_2.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy'])
model_2.summary()

model_2.fit(d, labels, batch_size=1024, epochs=10)
#model 3
hp = 0.25
inputs = Input(shape=(81, 100))
x = LSTM(1024, input_shape = (81, 100))(inputs)
x = Dropout(hp)(x)

x_sh = x
x = Dense(1024)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
#x = Dropout(hp)(x)
x = Add()([x, x_sh])
x = Activation('relu')(x)

x = Dense(256)(x)
x_sh = x
x = Dense(256)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
#x = Dropout(hp)(x)
x = Add()([x, x_sh])
x = Activation('relu')(x)

pre = Dense(12, activation='softmax')(x)
model_3 = Model(input=inputs, output=pre)
model_3.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])


model_3.summary()
model_3.fit(d, labels, batch_size=1024, epochs=10)
#model 4
hp = 0.2
inputs = Input(shape=(81, 100))
x = LSTM(1024, input_shape = (81, 100))(inputs)
x = Dropout(hp)(x)

x_sh = x
x = Dense(1024)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Add()([x, x_sh])
x = Activation('relu')(x)

x = Dense(256)(x)
x_sh = x
x = Dense(256)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = Dense(128)(x)
x_sh = x
x = Dense(128)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

#x = Dropout(hp)(x)
x = Add()([x, x_sh])
x = Activation('relu')(x)
pre = Dense(12, activation='softmax')(x)
model_4 = Model(input=inputs, output=pre)
model_4.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])


model_4.summary()
model_4.fit(d, labels, batch_size=1024, epochs=20)
