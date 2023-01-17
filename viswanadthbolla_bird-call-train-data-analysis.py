from plotly.subplots import make_subplots 

import plotly.graph_objects as go

import matplotlib.pyplot as plt

import IPython.display as ipd

import plotly.express as px

import librosa.display

import pandas as pd

import numpy as  np

import librosa

import warnings

import IPython

import os

os.listdir('../input/birdsong-recognition')
warnings.filterwarnings(action='ignore')
train = pd.read_csv('../input/birdsong-recognition/train.csv',)

train.head()

train.shape
train.isnull().sum()
col=train.columns

col
print(train['ebird_code'].value_counts().max())

print(train['ebird_code'].value_counts().min())
plt.figure(figsize=(10, 6))

train['ebird_code'].value_counts().plot(kind='hist',bins=100)

plt.grid()

plt.xlabel('no.of.examples')

plt.ylabel('no.of.species')

plt.show()



train.ebird_code.value_counts()
ratings=train.rating.unique().tolist()

ratings.sort()

ratings
plt.figure(figsize=(10, 6))

plt.hist(train.rating,rwidth=0.9)

plt.show()
train['playback_used'].fillna('Not Defined',inplace=True)

train.playback_used.value_counts()
train.channels.value_counts()
train.date
plt.figure(figsize=(20, 8))

train.date.value_counts().sort_index().plot()




train['date'].sort_values()[15:30].values







# Convert string to datetime64

train['date'] = train['date'].apply(pd.to_datetime,format='%Y-%m-%d', errors='coerce')

#train.set_index('date',inplace=True)

train['date'].value_counts().plot(figsize=(12,8))



train.pitch.value_counts()
train.duration.mode()
train.duration.median()
plt.figure(figsize=(20, 8))

train.duration.value_counts().sort_index().plot()

plt.xlabel('duration')

plt.ylabel('no.of.examples')

plt.grid()
train.filename.nunique()
train.filename.head()
train.speed.value_counts()
train.species.nunique()
train.species.value_counts()
train.number_of_notes.value_counts()
train.number_of_notes.value_counts().plot()
train.title
train.secondary_labels[0]
train.bird_seen.isnull().sum()
train['bird_seen'].fillna('Not Defined',inplace=True)
train.bird_seen.value_counts()
train.sci_name.nunique()
train.sci_name.value_counts()
train.location.nunique()
train.location.value_counts().sort_values()[6334:6349].plot(kind='barh')

plt.show()
train.latitude.nunique()
train.latitude.sample(10)
train.sampling_rate.value_counts()
train.type.nunique()
train.type.value_counts()[0:30]
train.elevation
df=pd.DataFrame()
df[['height','measurement','none']]= train.elevation.str.split(" ",expand=True,)
df.height.sample(20)
train.description.head(20)
train.bitrate_of_mp3.value_counts().index
train.file_type.value_counts()
train.volume.value_counts()
train.background.isnull().sum()
train.background.nunique()
train.background.value_counts()
train.xc_id.nunique()
train.xc_id.head(20)
train.url.nunique()
train.url.head()
train.country.value_counts()[0:10].plot(kind='barh')

plt.xlabel('no.of.examples')

plt.show()
train.author.nunique()
plt.figure(figsize=(12, 8))

train.author.value_counts()[0:20].plot(kind='barh')
train.primary_label.nunique()
train.primary_label
train.longitude.nunique()
train.longitude.sample(10)
train.length
train.length.value_counts()
train.time.nunique()
train.duration.head()
train.time.head()
train.recordist.nunique()
train.recordist.value_counts()
plt.figure(figsize=(12, 8))

train.recordist.value_counts()[0:20].plot(kind='barh')

plt.show()
train.license.nunique()
train.license.value_counts()