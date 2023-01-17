from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



dat = pd.read_csv('../input/kidney-catheter-data/kidney.csv')

dat.head()
sns.lmplot(x="time", y="age",  data=dat);
sns.lmplot(x="age", y="frail",  data=dat);
f, ax = plt.subplots(figsize=(6, 6))

sns.kdeplot(dat.age, dat.frail, ax=ax)

sns.rugplot(dat.age, color="g", ax=ax)

sns.rugplot(dat.frail, vertical=True, ax=ax);
f, ax = plt.subplots(figsize=(6, 6))

cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)

sns.kdeplot(dat.age, dat.frail, cmap=cmap, n_levels=60, shade=True);
sns.pairplot(dat, kind="reg")

plt.show()
sns.pairplot(dat, kind="scatter")

plt.show()
import seaborn as sns

plt.plot( 'age', 'frail', data=dat, marker='o', color='mediumvioletred')

plt.show()