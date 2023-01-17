from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



dat = pd.read_csv('../input/sugarcane-disease-data/cane.csv')

dat.head()
sns.lmplot(x="n", y="r", hue="block", data=dat)
sns.lmplot(x="n", y="x", hue="block", data=dat)
sns.lmplot(x="r", y="x", hue="block", data=dat)
f, ax = plt.subplots(figsize=(6, 6))

sns.kdeplot(dat.n, dat.r, ax=ax)

sns.rugplot(dat.n, color="g", ax=ax)

sns.rugplot(dat.r, vertical=True, ax=ax);
f, ax = plt.subplots(figsize=(6, 6))

cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)

sns.kdeplot(dat.n, dat.r, cmap=cmap, n_levels=60, shade=True);
g = sns.jointplot(x="n", y="r", data=dat, kind="kde", color="m")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)

g.set_axis_labels("$n$", "$r$");
f, ax = plt.subplots(figsize=(6, 6))

sns.kdeplot(dat.n, dat.x, ax=ax)

sns.rugplot(dat.n, color="g", ax=ax)

sns.rugplot(dat.x, vertical=True, ax=ax);
f, ax = plt.subplots(figsize=(6, 6))

cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)

sns.kdeplot(dat.n, dat.x, cmap=cmap, n_levels=60, shade=True);
g = sns.jointplot(x="n", y="x", data=dat, kind="kde", color="m")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)

g.set_axis_labels("$n$", "$x$");
f, ax = plt.subplots(figsize=(6, 6))

sns.kdeplot(dat.r, dat.x, ax=ax)

sns.rugplot(dat.r, color="g", ax=ax)

sns.rugplot(dat.x, vertical=True, ax=ax);
f, ax = plt.subplots(figsize=(6, 6))

cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)

sns.kdeplot(dat.r, dat.x, cmap=cmap, n_levels=60, shade=True);
g = sns.jointplot(x="r", y="x", data=dat, kind="kde", color="m")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)

g.set_axis_labels("$r$", "$x$");