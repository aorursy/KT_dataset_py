import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
%matplotlib inline

# Don't Show Warning Messages
import warnings
warnings.filterwarnings('ignore')
# demographic_info.txt was added in version 2

os.listdir('../input')
os.listdir('../input/respiratory_sound_database/Respiratory_Sound_Database')
# Install the pydub library

# Check that kernel Internet is connected before running this cell
! pip install pydub
# Play an audio file

from pydub import AudioSegment
import IPython

# We will listen to this file:
# 213_1p5_Pr_mc_AKGC417L.wav

audio_file = '213_1p5_Pr_mc_AKGC417L.wav'

path = \
'../input/respiratory_sound_database/Respiratory_Sound_Database/audio_and_txt_files/' + audio_file

IPython.display.Audio(path)
path ='../input/demographic_info.txt'
col_names = ['patient_id', 'age', 'sex', 'adult_bmi', 'child_weight', 'child_height']

# Adult BMI (kg/m2)
# Child Weight (kg)
# Child Height (cm)

df_demo = pd.read_csv(path, sep=" ", header=None, names=col_names)

df_demo.head(10)
path = \
'../input/respiratory_sound_database/Respiratory_Sound_Database/patient_diagnosis.csv'

df_diag = pd.read_csv(path, header=None, names=['patient_id', 'diagnosis'])

df_diag.head(10)
path = \
'../input/respiratory_sound_database/Respiratory_Sound_Database/filename_differences.txt'

df_diff = pd.read_csv(path, sep=" ", header=None, names=['file_names'])

df_diff.head(10)
path = \
'../input/respiratory_sound_database/Respiratory_Sound_Database/filename_format.txt'

data = open(path, 'r').read()

data
path = \
'../input/respiratory_sound_database/Respiratory_Sound_Database/audio_and_txt_files'

os.listdir(path)
path = \
'../input/respiratory_sound_database/Respiratory_Sound_Database/audio_and_txt_files/154_2b4_Al_mc_AKGC417L.txt'

col_names = ['Beginning_of_respiratory_cycle', 'End_of_respiratory_cycle', 'Presence/absence_of_crackles', 'Presence/absence_of_wheezes']

# Respiratory cycle column values are in 'seconds'.
# Presence = 1
# Absence = 0

df_annot = pd.read_csv(path, sep="\t", header=None, names=col_names)

df_annot.head(10)
path = \
'../input/respiratory_sound_database/Respiratory_Sound_Database/audio_and_txt_files/154_2b4_Al_mc_AKGC417L.txt'

col_names = ['Beginning_of_respiratory_cycle', 'End_of_respiratory_cycle', 'Presence/absence_of_crackles', 'Presence/absence_of_wheezes']

# Respiratory cycle column values are in 'seconds'.
# Presence = 1
# Absence = 0

df_annot = pd.read_csv(path, sep="\t", header=None, names=col_names)

df_annot.head(20)
# Install the pysoundfile library
! pip install pysoundfile
import soundfile as sf

# Define helper functions

# Load a .wav file. 
# These are 24 bit files. The PySoundFile library is able to read 24 bit files.
# https://pysoundfile.readthedocs.io/en/0.9.0/

def get_wav_info(wav_file):
    data, rate = sf.read(wav_file)
    return data, rate

# source: Andrew Ng Deep Learning Specialization, Course 5
def graph_spectrogram(wav_file):
    data, rate = get_wav_info(wav_file)
    nfft = 200 # Length of each window segment
    fs = 8000 # Sampling frequencies
    noverlap = 120 # Overlap between windows
    nchannels = data.ndim
    if nchannels == 1:
        pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap = noverlap)
    elif nchannels == 2:
        pxx, freqs, bins, im = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)
    return pxx
# plot the spectrogram

path = \
'../input/respiratory_sound_database/Respiratory_Sound_Database/audio_and_txt_files/154_2b4_Al_mc_AKGC417L.wav'


x = graph_spectrogram(path)
# choose an audio file
audio_file = '154_2b4_Al_mc_AKGC417L.wav'

path = \
'../input/respiratory_sound_database/Respiratory_Sound_Database/audio_and_txt_files/' + audio_file

# read the file
data, rate = sf.read(path)

# display the numpy array
data
# https://stackoverflow.com/questions/37999150/
# python-how-to-split-a-wav-file-into-multiple-wav-files

from pydub import AudioSegment

# note: Time is given in seconds. Will be converted to milliseconds later.
start_time = 0
end_time = 7

t1 = start_time * 1000 # pydub works in milliseconds
t2 = end_time * 1000
newAudio = AudioSegment.from_wav(path) # path is defined above
newAudio = newAudio[t1:t2]
newAudio.export('new_slice.wav', format="wav")
# Lets listen to the new slice

IPython.display.Audio('new_slice.wav')
