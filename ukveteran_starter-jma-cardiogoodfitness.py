from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



dat = pd.read_csv('../input/cardiogoodfitness/CardioGoodFitness.csv')
dat.head()
dat.describe()
p = dat.hist(figsize = (20,20))
plt.matshow(dat.corr())

plt.colorbar()

plt.show()
sns.countplot(x=dat['Miles'])

fig=plt.gcf()

fig.set_size_inches(6,4)
plt.scatter(dat['Income'], dat['Miles'])
sns.regplot(x=dat['Income'], y=dat['Miles'])
sns.lmplot(x="Income", y="Miles", hue="Gender", data=dat);
sns.lmplot(x="Income", y="Miles", hue="MaritalStatus", data=dat);