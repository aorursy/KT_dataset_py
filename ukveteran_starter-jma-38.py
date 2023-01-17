from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



test = pd.read_csv('../input/test_geom.csv')

train = pd.read_csv('../input/train_geom.csv')
test.head()
test.describe()
plt.matshow(test.corr())

plt.colorbar()

plt.show()
sns.lineplot(x='atom_index_0', y='atom_index_1', data=test)
p = test.hist(figsize = (20,20))
train.head()
train.describe()
plt.matshow(train.corr())

plt.colorbar()

plt.show()
sns.lineplot(x='atom_index_0', y='atom_index_1', data=train)
q = train.hist(figsize = (20,20))