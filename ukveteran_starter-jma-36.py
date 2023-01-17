from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



ff= pd.read_csv('../input/forestfires.csv')
ff.head()
ff.describe()
plt.matshow(ff.corr())

plt.colorbar()

plt.show()
plt.scatter(ff['wind'], ff['rain'])
sns.regplot(ff['wind'], ff['rain'])
sns.lineplot(x='wind', y='rain', data=ff)
plt.style.use('fast')

sns.jointplot(x='wind', y='rain', data=ff)

plt.show()
q = sns.boxenplot(ff['wind'], ff['rain'], palette = 'rocket')
p = ff.hist(figsize = (20,20))