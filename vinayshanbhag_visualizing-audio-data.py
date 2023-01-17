# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



import scipy.io.wavfile as wav

import scipy.signal as signal

import matplotlib.pyplot as plt

import seaborn as sns

import os

import numpy as np

%matplotlib inline
!ls ../input/Training
files = sorted(os.listdir("../input/Training"))

columns=5

fig, ax = plt.subplots(int(np.ceil(len(files)/columns))*2,columns,figsize=(15,30))

fig.suptitle("Frequency Spectrum & Oscillogram", x=0.5, y=0.91, fontsize=16)

for idx, file in enumerate(files):

    r,c = idx//columns*2, idx%columns

    rate, data = wav.read("../input/Training/{}".format(file))

    f, t, Sxx = signal.spectrogram(data, fs=rate)

    d = 20*np.log10(Sxx+1e-10)

    ax[r,c].pcolormesh(t,f,d, vmin=-1e1,vmax=d.max())

    ax[r,c].set_title(file);

    if not c and not r:

        ax[r,c].set_xlabel("time")

        ax[r,c].set_ylabel("frequency");

        ax[r,c].set_xticks([])

        ax[r,c].set_frame_on(False)

        ax[r,c].set_yticks([])

    else: ax[r,c].axis("off")

    

    norm_data = (data -data.mean())/data.std()

    ax[r+1,c].plot(norm_data,lw=0.03)

    ax[r+1,c].axis("off") 



plt.subplots_adjust(wspace=0.05, hspace=0.1)
fig, ax = plt.subplots(int(np.ceil(len(files)/columns)),columns,figsize=(10,10))

fig.suptitle("Spectral Power Density", x=0.5, y=1.05, fontsize=16)

for idx, file in enumerate(files):

    r,c = idx//columns, idx%columns

    rate, data = wav.read("../input/Training/{}".format(file))

    f, Pxx = signal.welch(data, fs=rate)

    ax[r,c].semilogy(f,Pxx)

    ax[r,c].set_title(file);

    if not c and not r:

        ax[r,c].set_xlabel("frequency")

        ax[r,c].set_ylabel("power");

    

plt.tight_layout()