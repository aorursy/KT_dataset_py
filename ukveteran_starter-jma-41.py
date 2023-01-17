from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



dat = pd.read_csv('../input/weatherAUS.csv')
dat.head()
dat.describe()
plt.matshow(dat.corr())

plt.colorbar()

plt.show()
p = dat.hist(figsize = (20,20))
sns.lineplot(x='MinTemp', y='MaxTemp', data=dat)
sns.regplot(dat['MinTemp'], dat['MaxTemp'])