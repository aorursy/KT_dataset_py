from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



dat = pd.read_csv('../input/wine_dataset.csv')
dat.head()
dat.describe()
plt.matshow(dat.corr())

plt.colorbar()

plt.show()
plt.scatter(dat['citric_acid'], dat['residual_sugar'])
sns.lineplot(x='citric_acid', y='residual_sugar', data=dat)
plt.style.use('fast')

sns.jointplot(x='citric_acid', y='residual_sugar', data=dat)

plt.show()