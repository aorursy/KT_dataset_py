from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



dat = pd.read_csv('../input/LOTTOMAX.csv')
dat.head()
dat.describe()
plt.matshow(dat.corr())

plt.colorbar()

plt.show()
plt.scatter(dat['NUMBER DRAWN 1'], dat['BONUS NUMBER'])
sns.regplot(x=dat['NUMBER DRAWN 1'], y=dat['BONUS NUMBER'])
sns.lineplot(x='NUMBER DRAWN 1', y='BONUS NUMBER', data=dat)
plt.style.use('fast')

sns.jointplot(x='NUMBER DRAWN 1', y='BONUS NUMBER', data=dat)

plt.show()
q1 = sns.boxenplot(x = dat['NUMBER DRAWN 1'], y = dat['BONUS NUMBER'], palette = 'rocket')