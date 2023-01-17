from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



dat = pd.read_csv('../input/data.csv')
dat.head()
dat.describe()
plt.matshow(dat.corr())

plt.colorbar()

plt.show()
p = dat.hist(figsize = (20,20))
sns.lineplot(x='radius_mean', y='area_mean', data=dat)
sns.lineplot(x='radius_mean', y='perimeter_mean', data=dat)
sns.regplot(dat['radius_mean'], dat['perimeter_mean'])
sns.regplot(dat['radius_mean'], dat['area_mean'])