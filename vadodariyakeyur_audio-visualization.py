# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
'''

LibROSA is a python package that helps us analyse audio files and provides the building blocks necessary to 

create audio information retrieval systems.

'''



!pip install librosa
import IPython.display as ipd



import librosa



import librosa.display



import matplotlib.pyplot as plt
'''

Loading And Playing Audio Files In Jupyter

'''



ipd.Audio('/kaggle/input/respiratory-sound-database/respiratory_sound_database/Respiratory_Sound_Database/audio_and_txt_files/165_1b1_Ar_sc_Meditron.wav')
'''

Initialize the plot with a figure size.

'''



plt.figure(figsize=(15,4))
'''

We will then load the audio file using librosa and will collect the data array 

and sampling rate for the audio file.

'''



filename = '/kaggle/input/respiratory-sound-database/respiratory_sound_database/Respiratory_Sound_Database/audio_and_txt_files/165_1b1_Ar_sc_Meditron.wav'



data,sample_rate1 = librosa.load(filename, sr=22050, mono=True, offset=0.0, duration=50, res_type='kaiser_best')
'''

The sampling rate is the number of samples per second. Hz or Hertz is the unit of the sampling rate.

20 kHz is the audible range for human beings.

'''



librosa.display.waveplot(data,sr=sample_rate1, max_points=50000.0, x_axis='time', offset=0.0, max_sr=1000)
y, sr = librosa.load(filename)

librosa.feature.melspectrogram(y=y, sr=sr)
D = np.abs(librosa.stft(y))**2

S = librosa.feature.melspectrogram(S=D, sr=sr)
# Passing through arguments to the Mel filters

S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,

                                    fmax=8000)
fig, ax = plt.subplots()

S_dB = librosa.power_to_db(S, ref=np.max)

img = librosa.display.specshow(S_dB, x_axis='time',

                         y_axis='mel', sr=sr,

                         fmax=8000, ax=ax)

fig.colorbar(img, ax=ax, format='%+2.0f dB')

ax.set(title='Mel-frequency spectrogram')