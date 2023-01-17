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
import json, sys, random

import numpy as np
# download dataset from json object

f = open(r'../input/ships-in-satellite-imagery/shipsnet.json')

dataset = json.load(f)

f.close()
data = np.array(dataset['data']).astype('uint8')

labels = np.array(dataset['labels']).astype('uint8')
data.shape
print(labels.shape)

print(labels[0:10])
n_spectrum = 3 # color chanel (RGB)

width = 80

height = 80

X = data.reshape([-1, n_spectrum, width, height])



X.shape
# Your code here...

import matplotlib.pyplot as plt

for i in range(20):

    plt.imshow(np.moveaxis(X[i], 0, -1)) # reshape won't work

    plt.show()
noships = np.argwhere(labels==0)[:20].flatten()

for i in noships:

    plt.imshow(np.moveaxis(X[i], 0, -1))

    plt.show()
# Your code here...



Xmeans = X.mean(axis=0) # returns array of shape (n_spectrum, width(?), height(?))

Xstds = X.std(axis=0)



# Means for the three channels

fig, axs = plt.subplots(1, 3, constrained_layout=True)

titles = list('RGB')

for i, ax in enumerate(axs):

    ax.hist(Xmeans[i, :, :].flatten())

    ax.set_title(titles[i])

fig.suptitle('means')

fig.show()



# Std for the three channels

fig, axs = plt.subplots(1, 3, constrained_layout=True)

for i, ax in enumerate(axs):

    ax.hist(Xstds[i, :, :].flatten())

    ax.set_title(titles[i])

fig.suptitle('stds')

fig.show()
# Your code here...