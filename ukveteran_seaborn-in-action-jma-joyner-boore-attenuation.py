from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



dat = pd.read_csv('../input/the-joynerboore-attenuation-data/attenu.csv')

dat.head()
sns.lmplot(x="dist", y="accel", hue="mag", data=dat)
f, ax = plt.subplots(figsize=(6, 6))

sns.kdeplot(dat.dist, dat.accel, ax=ax)

sns.rugplot(dat.dist, color="g", ax=ax)

sns.rugplot(dat.accel, vertical=True, ax=ax);
f, ax = plt.subplots(figsize=(6, 6))

cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)

sns.kdeplot(dat.dist, dat.accel, cmap=cmap, n_levels=60, shade=True);
g = sns.jointplot(x="dist", y="accel", data=dat, kind="kde", color="m")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)

g.set_axis_labels("$dist$", "$accel$");