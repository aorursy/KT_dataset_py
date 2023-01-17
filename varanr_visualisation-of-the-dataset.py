# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

%matplotlib inline
import json
#change file name here

with open('../input/abstraction-and-reasoning-challenge/training/178fcbfb.json') as f:

    file = json.load(f)

    

total = len(file['train'])

  

plt.figure(num=None, figsize=(10,15))

p = 1

for exp in file['train']:

    inp = np.array(exp['input'])

    out = np.array(exp['output'])

    for val in [inp, out]:

        plt.subplot(total,2,p)

        p += 1

        plt.imshow(val)