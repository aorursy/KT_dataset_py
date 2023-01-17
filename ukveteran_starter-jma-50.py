from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



dat = pd.read_csv('../input/sorteios.csv')
dat.head()
dat.describe()
plt.matshow(dat.corr())

plt.colorbar()

plt.show()
plt.scatter(dat['1ª Dezena'], dat['4ª Dezena'])
sns.regplot(x=dat['1ª Dezena'], y=dat['4ª Dezena'])
sns.lineplot(x='1ª Dezena', y='4ª Dezena', data=dat)
plt.style.use('fast')

sns.jointplot(x='1ª Dezena', y='4ª Dezena', data=dat)

plt.show()
q1 = sns.boxenplot(x = dat['1ª Dezena'], y = dat['4ª Dezena'], palette = 'rocket')