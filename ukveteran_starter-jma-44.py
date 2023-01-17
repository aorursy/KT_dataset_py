from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



dat = pd.read_csv('../input/100mesh.csv')
dat.head()
dat.describe()
plt.matshow(dat.corr())

plt.colorbar()

plt.show()
p = dat.hist(figsize = (20,20))
sns.lineplot(x='theta_deg', y='sintheta', data=dat)
sns.regplot(dat['theta_deg'], dat['sintheta'])
sns.regplot(dat['sintheta'], dat['sin_squared_theta'])