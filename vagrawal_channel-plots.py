import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import glob

from scipy.io import loadmat
files = glob.glob('../input/train_1/*.mat');
for channel in range(16):

    f = plt.figure(figsize=(12,2))

    for file in files[:100]:

        data = np.swapaxes(loadmat(file)['dataStruct']['data'].item(0).astype(np.float32), 0, 1)[channel]

        plt.plot(data.reshape((10000, 24)).mean(0), color = 'blue' if file[-5] == '0' else 'red', alpha = 0.3)