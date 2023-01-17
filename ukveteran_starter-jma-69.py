from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



dat = pd.read_csv('../input/open-units/open_units.csv')
dat.head()
dat.describe()
plt.matshow(dat.corr())

plt.colorbar()

plt.show()
plt.scatter(dat['Units (4 Decimal Places)'], dat['Units per 100ml'])
sns.regplot(x=dat['Units (4 Decimal Places)'], y=dat['Units per 100ml'])
plt.style.use('fast')

sns.jointplot(x='Units (4 Decimal Places)', y='Units per 100ml', data=dat)

plt.show()
sns.lineplot(x='Units (4 Decimal Places)', y='Units per 100ml', data=dat)
p = dat.hist(figsize = (20,20))
ax = sns.violinplot(x=dat["Units per 100ml"])
ax = sns.violinplot(x=dat["Quantity"])
ax = sns.violinplot(x=dat["ABV"])
ax = sns.violinplot(y=dat["ABV"])