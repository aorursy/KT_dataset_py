# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import pickle

from time import time
## loading vectors in text format



t = time()

df = pd.read_csv('../input/glove6b50dtxt/glove.6B.50d.txt', sep=" ", quoting=3, header=None, index_col=0)

glove = {key: val.values for key, val in df.T.items()}

print(time()-t)
## saving vectors in pickle (binary) format



t = time()

with open('glove.6B.50d.pkl', 'wb') as fp:

    pickle.dump(glove, fp)

print(time()-t)
len(glove)
## loading vectors in binary format



t = time()

with open('glove.6B.50d.pkl', 'rb') as fp:

    glove = pickle.load(fp)

print(time()-t)