### Importing libraries
import numpy as np
import pandas as pd

import matplotlib.colors as colors
import matplotlib.pyplot as plt

%matplotlib inline

import scipy.signal as signal
import os
import gc
## Listing the files in the directory
print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv', skiprows=range(1,4000000), nrows=2000000, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
## Checking the number of samples in the file
print("Dataset size:", train.shape)
## Showing the first data samples with high precision
pd.set_option("display.precision", 10)
train.head()
(1/4e6)*1e7
def calcSTFT_norm(inputSignal, samplingFreq, window='hann', nperseg=256, figsize=(9,5), cmap='magma', ylim_max=None):
    '''Calculates the STFT for a time series:
        inputSignal: numpy array for the signal (it also works for Pandas.Series);
        samplingFreq: the sampling frequency;
        window : str or tuple or array_like, optional
            Desired window to use. If `window` is a string or tuple, it is
            passed to `get_window` to generate the window values, which are
            DFT-even by default. See `get_window` for a list of windows and
            required parameters. If `window` is array_like it will be used
            directly as the window and its length must be nperseg. Defaults
            to a Hann window.
        nperseg : int, optional
            Length of each segment. Defaults to 256.
        figsize: the plot size, set as (6,3) by default;
        cmap: the color map, set as the divergence Red-Yellow-Green by default;
        ylim_max: the max frequency to be shown. By default it's the half sampling frequency.'''
    ##Calculating STFT
    f, t, Zxx = signal.stft(inputSignal, samplingFreq, window=window, nperseg=nperseg)
    ##Plotting STFT
    fig = plt.figure(figsize=figsize)
    ### Different methods can be chosen for normalization: PowerNorm; LogNorm; SymLogNorm.
    ### Reference: https://matplotlib.org/tutorials/colors/colormapnorms.html
    spec = plt.pcolormesh(t, f, np.abs(Zxx), 
                          norm=colors.PowerNorm(gamma=1./8.),
                          #norm=colors.LogNorm(vmin=np.abs(Zxx).min(), vmax=np.abs(Zxx).max()),
                          #norm=colors.SymLogNorm(linthresh=0.13, linscale=1,
                          #                       vmin=-1.0, vmax=1.0),
                          cmap=plt.get_cmap(cmap))
    cbar = plt.colorbar(spec)
    ##Plot adjustments
    plt.title('STFT Spectrogram')
    ax = fig.axes[0]
    ax.grid(True)
    ax.set_title('STFT Magnitude')
    if ylim_max:
        ax.set_ylim(0,ylim_max)
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')
    fig.show
    return
calcSTFT_norm(train['acoustic_data'], 4e6, nperseg=1048576, ylim_max=300000)
## Creating a list of positions where the "time_to_failure" is null
failure_regions = {}
## Defining a function to find the local minima positions
def is_failure(df):
    failures = df.index[(df.time_to_failure.shift(1) > df.time_to_failure) & (df.time_to_failure.shift(-1) > df.time_to_failure)].tolist()
    return failures
## Returns a signal region for a given failure position in a pandas.chunk:
## REMEMBER: a chunk is a pandas.Dataframe.
def region_readChunk(pos, chunk):
    ## Setting the region limits:
    beforePos = pos - 4e6
    afterPos = pos + 1e6
    #In some cases it can be shorter than the other regions
    if beforePos < chunk.index.min():
        beforePos = chunk.index.min()
    if afterPos > chunk.index.max():
        afterPos = chunk.index.max()
    ## Extracting the region data:
    data = chunk.loc[int(beforePos):int(afterPos)]
    return data
## Processing the dataset in chunks:
## For each chunk, find the local minima and extract the respective region.
for chunk in pd.read_csv('../input/train.csv', chunksize=5e7, 
                         dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64}):
    failure_pos = is_failure(chunk)
    for pos in failure_pos:
        data = region_readChunk(pos, chunk)
        failure_regions[pos] = data
## Garbage collection (to reduce used memory)
gc.collect()
failure_regions.keys()
posFailures = pd.DataFrame(list(failure_regions.keys()))
posFailures.describe()
def region_plot(df):
    data = df.copy()
    ##Aplying a 1e3 gain to the "time_to_failure" in order to make the failure visible
    data['time_to_failure'] = data['time_to_failure']*100
    ##Calculate elapsed time through the index and sampling frequency (4e6 MHz)
    data['time'] = data.index
    data['time'] = data['time']*(1/4e6)
    ##Normalize time information (0.0s  to 1.25s)
    data['Time [sec]'] = data['time'] - data['time'].min()
    ##Plot lines
    data[['acoustic_data','time_to_failure','Time [sec]']].plot(x='Time [sec]', figsize=(8,5))
    return
#for key, value in failure_regions.items():
#    region_plot(value)
region_plot(failure_regions[419368879])
calcSTFT_norm(failure_regions[419368879]['acoustic_data'], 4e6, nperseg=1048576, ylim_max=300000)
