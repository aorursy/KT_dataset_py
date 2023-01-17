from scipy.io import loadmat

from scipy.fftpack import fft, fftfreq

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
ictal_class_dict = {'interictal': 0, 'preictal': 1}

def load_sample(patient, segment, ictal_class):

    """

    Load sample from training set 1

    

    df = load_sample(patient=1, segment=1, ictal_class='interictal')

    """

    mat = loadmat('../input/train_1/{0}_{1}_{2}.mat'.format(

        patient, segment, ictal_class_dict[ictal_class]))

    mdata = mat['dataStruct']

    mtype = mdata.dtype

    ndata = {n: mdata[n][0,0] for n in mtype.names}

    data_headline = ndata['channelIndices'][0]

    data_raw = ndata['data']

    pdata = pd.DataFrame(data_raw,columns=data_headline)

    iEEGsamplingRate = ndata['iEEGsamplingRate']

    nSamplesSegment = int(ndata['nSamplesSegment'][0,0])

    return pdata, iEEGsamplingRate, nSamplesSegment
interictal, r, n = load_sample(patient=1, segment=1, ictal_class='interictal')
interictal[[1.0, 2.0]].plot(subplots=True)
r, n
preictal, r, n = load_sample(patient=1, segment=1, ictal_class='preictal')
preictal[[1.0, 2.0]].plot(subplots=True)
r, n
from scipy.fftpack import fft, fftfreq
freqs = fftfreq(n, 1/r)
pre1 = preictal[1.0]

ft_pre1_values = fft(pre1)
len(ft_pre1_values)
freqs[0]
ft_pre1 = pd.Series(data=ft_pre1_values, index=freqs[0])
preabs = np.abs(ft_pre1)
presum = np.abs(np.sqrt(np.sum(ft_pre1**2))) # abs since it is a complex number

prerel = preabs / presum
prerel.plot(logy=True)

plt.ylim([0, 0.1])
prerel.hist(bins=100)