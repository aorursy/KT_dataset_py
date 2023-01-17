from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



dat = pd.read_csv('../input/abalone.csv')
dat.head()
dat.describe()
plt.matshow(dat.corr())

plt.colorbar()

plt.show()
p = dat.hist(figsize = (20,20))
sns.lineplot(x='Diameter', y='Height', data=dat)
sns.regplot(dat['Diameter'], dat['Height'])
sns.lineplot(x='Diameter', y='Rings', data=dat)
sns.regplot(dat['Diameter'], dat['Rings'])