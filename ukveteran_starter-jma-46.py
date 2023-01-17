from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



dat = pd.read_csv('../input/pb_winning_numbers_03-18-2017.csv')
dat.head()
dat.describe()
plt.matshow(dat.corr())

plt.colorbar()

plt.show()
plt.scatter(dat['PB'], dat['PP'])
sns.regplot(x=dat['PB'], y=dat['PP'])
sns.lineplot(x='PB', y='PP', data=dat)
plt.style.use('fast')

sns.jointplot(x = 'PB', y = 'PP', data = dat)

plt.show()
q1 = sns.boxenplot(x = dat['PB'], y = dat['PP'], palette = 'rocket')