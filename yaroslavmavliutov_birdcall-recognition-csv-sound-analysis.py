import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from IPython.display import Audio
import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import seaborn as sns


import torch
import torch.nn as nn
from torch.optim import Adam
from albumentations import Normalize
from torchvision.models import resnet34
from torch.utils.data import Dataset, DataLoader
from torch import FloatTensor, LongTensor, DoubleTensor

from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences as pad
import warnings
warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True
train_csv = pd.read_csv("../input/birdsong-recognition/train.csv")
test_csv = pd.read_csv("../input/birdsong-recognition/test.csv")
print("There are {:,} unique bird species in the dataset.".format(len(train_csv['species'].unique())))
train_csv.head(3)
test_csv.head()
example_audio = '../input/birdsong-recognition/train_audio/amebit/XC127371.mp3'
y, sr = librosa.load(example_audio, sr=None)
print("Class:", example_audio.split('/')[-2])
Audio(example_audio)
train_csv['year'] = train_csv['date'].apply(lambda x: x.split('-')[0])
train_csv['month'] = train_csv['date'].apply(lambda x: x.split('-')[1])
plt.figure(figsize=(12, 5))
ax = sns.countplot(train_csv['date'].apply(lambda x: x.split('-')[0]), palette="hls")
plt.title("Audio Files Registration per Year", fontsize=12)
plt.xticks(rotation=90, fontsize=13)
plt.ylabel('Frequency')
plt.xlabel('Year')
plt.xlabel("");

plt.figure(figsize=(12, 5))
ax = sns.countplot(train_csv['date'].apply(lambda x: x.split('-')[1]), palette="hls")
plt.title("Audio Files Registration per Month", fontsize=12)
plt.xticks(rotation=90, fontsize=13)
plt.ylabel('Frequency')
plt.xlabel('Month')
plt.xlabel("");
top15 = list(train_csv['country'].value_counts().head(15).reset_index()['index'])
data = train_csv[train_csv['country'].isin(top15)]

plt.figure(figsize=(12, 5))
ax = sns.countplot(data['country'], palette='hls', order = data['country'].value_counts().index)

plt.title("Top 15 Countries with most Recordings", fontsize=16)
plt.ylabel("Frequency", fontsize=14)
plt.yticks(fontsize=13)
plt.xticks(rotation=45, fontsize=13)
plt.xlabel("");
train_csv['duration_interval'] = ">500"
train_csv.loc[train_csv['duration'] <= 100, 'duration_interval'] = "<=100"
train_csv.loc[(train_csv['duration'] > 100) & (train_csv['duration'] <= 200), 'duration_interval'] = "100-200"
train_csv.loc[(train_csv['duration'] > 200) & (train_csv['duration'] <= 300), 'duration_interval'] = "200-300"
train_csv.loc[(train_csv['duration'] > 300) & (train_csv['duration'] <= 400), 'duration_interval'] = "300-400"
train_csv.loc[(train_csv['duration'] > 400) & (train_csv['duration'] <= 500), 'duration_interval'] = "400-500"

plt.figure(figsize=(12, 5))
ax = sns.countplot(train_csv['duration_interval'], palette="hls")

plt.title("Distribution of Recordings Duration", fontsize=16)
plt.ylabel("Frequency", fontsize=14)
plt.yticks(fontsize=13)
plt.xticks(rotation=45, fontsize=13)
plt.xlabel("");
top7birds = list(train_csv['ebird_code'].value_counts().head(7).reset_index()['index'])
print(top7birds)
birds = dict()
for bird in top7birds:
    birds[bird] = '../input/birdsong-recognition/train_audio/' + bird + '/' + train_csv[train_csv['ebird_code'] == bird].sample(1, random_state = 33)['filename'].values[0]
ipd.Audio(list(birds.values())[0])
ipd.Audio(list(birds.values())[1])
ipd.Audio(list(birds.values())[2])
ipd.Audio(list(birds.values())[3])
ipd.Audio(list(birds.values())[4])
ipd.Audio(list(birds.values())[5])
ipd.Audio(list(birds.values())[6])
birds_audio = dict()
for bird in birds.keys():
    y, sr = librosa.load(birds[bird])
    birds_audio[bird], _ = librosa.effects.trim(y)
fig, ax = plt.subplots(len(top7birds), figsize = (16, 9))
fig.suptitle('Sound Waves', fontsize=16)

for i, bird in zip(range(len(top7birds)), top7birds):
    librosa.display.waveplot(y = birds_audio[bird], sr = sr, ax=ax[i])
    ax[i].set_ylabel(bird, fontsize=13)