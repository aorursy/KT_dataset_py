from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



dat = pd.read_csv('../input/highway-accidents/Highway1.csv')

dat.head()
f, ax = plt.subplots(figsize=(6, 6))

sns.kdeplot(dat.rate, dat.len, ax=ax)

sns.rugplot(dat.rate, color="g", ax=ax)

sns.rugplot(dat.len, vertical=True, ax=ax);
f, ax = plt.subplots(figsize=(6, 6))

cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)

sns.kdeplot(dat.rate, dat.len, cmap=cmap, n_levels=60, shade=True);
g = sns.jointplot(x="rate", y="len", data=dat, kind="kde", color="m")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)

g.set_axis_labels("$Rate$", "$Len$");
f, ax = plt.subplots(figsize=(6, 6))

sns.kdeplot(dat.adt, dat.trks, ax=ax)

sns.rugplot(dat.adt, color="g", ax=ax)

sns.rugplot(dat.trks, vertical=True, ax=ax);
f, ax = plt.subplots(figsize=(6, 6))

cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)

sns.kdeplot(dat.adt, dat.trks, cmap=cmap, n_levels=60, shade=True);
g = sns.jointplot(x="adt", y="trks", data=dat, kind="kde", color="m")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)

g.set_axis_labels("$Adt$", "$Trks$");