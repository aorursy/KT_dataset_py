from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



dat = pd.read_csv('../input/tipping/tips.csv')
dat.head()
dat.describe()
p = dat.hist(figsize = (20,20))
plt.matshow(dat.corr())

plt.colorbar()

plt.show()
sns.countplot(x=dat['sex'])

fig=plt.gcf()

fig.set_size_inches(6,4)
dat['size'].value_counts().plot(kind='bar', title='Size of the Party',figsize=(20,8)) 