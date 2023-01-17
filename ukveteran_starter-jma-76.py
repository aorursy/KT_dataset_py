from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



dat = pd.read_csv('../input/golden-gate-accel-20180512/Golden Gate Bridge Accelerometer Data.csv')
dat.head()
dat.describe()
p = dat.hist(figsize = (20,20))
plt.matshow(dat.corr())

plt.colorbar()

plt.show()
sns.regplot(x=dat['ax'], y=dat['ay'])
sns.regplot(x=dat['ax'], y=dat['az'])
sns.regplot(x=dat['ax'], y=dat['aT'])
sns.regplot(x=dat['ay'], y=dat['az'])
sns.regplot(x=dat['ay'], y=dat['aT'])
sns.regplot(x=dat['az'], y=dat['aT'])