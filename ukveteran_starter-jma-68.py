from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



dat = pd.read_csv('../input/median-listing-price-1-bedroom/median_price.csv')
dat.head()
dat.describe()
plt.matshow(dat.corr())

plt.colorbar()

plt.show()
plt.scatter(dat['2015-12'], dat['2016-01'])
sns.regplot(x=dat['2015-12'], y=dat['2016-01'])
plt.style.use('fast')

sns.jointplot(x='2015-12', y='2016-01', data=dat)

plt.show()
sns.lineplot(x='2015-12', y='2016-01', data=dat)