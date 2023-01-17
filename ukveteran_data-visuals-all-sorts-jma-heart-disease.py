from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



dat = pd.read_csv('../input/heart-disease-uci/heart.csv')

dat.head()
sns.lmplot(x="trestbps", y="chol",  data=dat);
f, ax = plt.subplots(figsize=(6, 6))

sns.kdeplot(dat.trestbps, dat.chol, ax=ax)

sns.rugplot(dat.trestbps, color="g", ax=ax)

sns.rugplot(dat.chol, vertical=True, ax=ax);
f, ax = plt.subplots(figsize=(6, 6))

cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)

sns.kdeplot(dat.trestbps, dat.chol, cmap=cmap, n_levels=60, shade=True);
sns.pairplot(dat, kind="reg")

plt.show()
sns.pairplot(dat, kind="scatter")

plt.show()
import seaborn as sns

plt.plot( 'trestbps', 'chol', data=dat, marker='o', color='mediumvioletred')

plt.show()