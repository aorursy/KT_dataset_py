from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



dat = pd.read_csv('../input/chemical-composition-of-pottery/Pottery.csv')

dat.head()
sns.lmplot(x="Al", y="Fe", hue="Site", data=dat);
sns.lmplot(x="Mg", y="Ca", hue="Site", data=dat);
f, ax = plt.subplots(figsize=(6, 6))

sns.kdeplot(dat.Al, dat.Ca, ax=ax)

sns.rugplot(dat.Al, color="g", ax=ax)

sns.rugplot(dat.Ca, vertical=True, ax=ax);
f, ax = plt.subplots(figsize=(6, 6))

sns.kdeplot(dat.Mg, dat.Na, ax=ax)

sns.rugplot(dat.Mg, color="g", ax=ax)

sns.rugplot(dat.Na, vertical=True, ax=ax);
f, ax = plt.subplots(figsize=(6, 6))

cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)

sns.kdeplot(dat.Al, dat.Ca, cmap=cmap, n_levels=60, shade=True);
f, ax = plt.subplots(figsize=(6, 6))

cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)

sns.kdeplot(dat.Mg, dat.Na, cmap=cmap, n_levels=60, shade=True);
g = sns.jointplot(x="Al", y="Ca", data=dat, kind="kde", color="m")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)

g.set_axis_labels("$Al$", "$Ca$");
g = sns.jointplot(x="Mg", y="Na", data=dat, kind="kde", color="m")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)

g.set_axis_labels("$Mg$", "$Na$");