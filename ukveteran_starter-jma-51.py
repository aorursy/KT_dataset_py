from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



dat = pd.read_csv('../input/wineanalysis.csv')
dat.head()
dat.describe()
plt.matshow(dat.corr())

plt.colorbar()

plt.show()
plt.scatter(dat['citric.acid'], dat['residual.sugar'])
sns.regplot(x=dat['citric.acid'], y=dat['residual.sugar'])
sns.lineplot(x='citric.acid', y='residual.sugar', data=dat)
plt.style.use('fast')

sns.jointplot(x='citric.acid', y='residual.sugar', data=dat)

plt.show()
q1 = sns.boxenplot(x=dat['citric.acid'], y=dat['residual.sugar'], palette = 'rocket')