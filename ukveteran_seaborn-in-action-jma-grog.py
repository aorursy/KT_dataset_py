from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



dat = pd.read_csv('../input/alcohol-consumption-in-australia-and-new-zealand/grog.csv')

dat.head()
sns.lmplot(x="Beer", y="Wine", hue="Country", data=dat);
sns.lmplot(x="Beer", y="Spirit", hue="Country", data=dat);
sns.lmplot(x="Wine", y="Spirit", hue="Country", data=dat);
f, ax = plt.subplots(figsize=(6, 6))

sns.kdeplot(dat.Beer, dat.Wine, ax=ax)

sns.rugplot(dat.Beer, color="g", ax=ax)

sns.rugplot(dat.Wine, vertical=True, ax=ax);
f, ax = plt.subplots(figsize=(6, 6))

sns.kdeplot(dat.Beer, dat.Spirit, ax=ax)

sns.rugplot(dat.Beer, color="g", ax=ax)

sns.rugplot(dat.Spirit, vertical=True, ax=ax);
f, ax = plt.subplots(figsize=(6, 6))

sns.kdeplot(dat.Wine, dat.Spirit, ax=ax)

sns.rugplot(dat.Wine, color="g", ax=ax)

sns.rugplot(dat.Spirit, vertical=True, ax=ax);
f, ax = plt.subplots(figsize=(6, 6))

cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)

sns.kdeplot(dat.Beer, dat.Wine, cmap=cmap, n_levels=60, shade=True);
f, ax = plt.subplots(figsize=(6, 6))

cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)

sns.kdeplot(dat.Beer, dat.Spirit, cmap=cmap, n_levels=60, shade=True);
f, ax = plt.subplots(figsize=(6, 6))

cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)

sns.kdeplot(dat.Wine, dat.Spirit, cmap=cmap, n_levels=60, shade=True);
g = sns.jointplot(x="Beer", y="Wine", data=dat, kind="kde", color="m")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)

g.set_axis_labels("$Beer$", "$Wine$");
g = sns.jointplot(x="Beer", y="Spirit", data=dat, kind="kde", color="m")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)

g.set_axis_labels("$Beer$", "$Spirit$");
g = sns.jointplot(x="Wine", y="Spirit", data=dat, kind="kde", color="m")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)

g.set_axis_labels("$Wine$", "$Spirit$");