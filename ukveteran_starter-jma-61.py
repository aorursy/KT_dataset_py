from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



dat = pd.read_csv('../input/avocado.csv')
dat.head()
dat.describe()
plt.matshow(dat.corr())

plt.colorbar()

plt.show()
plt.scatter(dat['AveragePrice'], dat['Total Volume'])
sns.regplot(x=dat['AveragePrice'], y=dat['Total Volume'])
sns.lineplot(x='AveragePrice', y='Total Volume', data=dat)