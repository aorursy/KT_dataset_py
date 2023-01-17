from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



dat = pd.read_csv('../input/cardiogoodfitness/CardioGoodFitness.csv')

dat.head()
p = dat.hist(figsize = (20,20))
sns.regplot(x=dat['Income'], y=dat['Miles'])
sns.lmplot(x="Income", y="Miles", hue="MaritalStatus", data=dat);
f, ax = plt.subplots(figsize=(6, 6))

sns.kdeplot(dat.Income, dat.Miles, ax=ax)

sns.rugplot(dat.Income, color="g", ax=ax)

sns.rugplot(dat.Miles, vertical=True, ax=ax);
f, ax = plt.subplots(figsize=(6, 6))

cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)

sns.kdeplot(dat.Income, dat.Miles, cmap=cmap, n_levels=60, shade=True);
g = sns.jointplot(x="Income", y="Miles", data=dat, kind="kde", color="m")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)

g.set_axis_labels("$Income$", "$Miles$");