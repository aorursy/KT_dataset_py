# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# load training and test data in panda data frames

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
# output top of train dataset

train_df.head()
print(train_df.columns.values)

print(train_df.values[:10])
ages = train_df.values[:, 5:6]

ages = [a[0] for a in ages if ~np.isnan(a[0])]

print(len(ages))



n, bins, patches = plt.hist(ages, 10, normed=1, facecolor='g', alpha=0.75)

plt.grid(True)

plt.show()

print(n, bins, patches)