from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



dat = pd.read_csv('../input/test.csv')
dat.head()
dat.describe()
plt.matshow(dat.corr())

plt.colorbar()

plt.show()
plt.scatter(dat['c1'], dat['s1'])
sns.regplot(x=dat['c1'], y=dat['c3'])
sns.lineplot(x='c1', y='c2', data=dat)
datt = pd.read_csv('../input/train.csv')
datt.head()
datt.describe()
plt.matshow(datt.corr())

plt.colorbar()

plt.show()
sns.lineplot(x='c1', y='c2', data=datt)
sns.regplot(x=datt['c1'], y=datt['c3'])