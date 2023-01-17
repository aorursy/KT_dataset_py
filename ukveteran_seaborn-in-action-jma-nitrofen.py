from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



dat = pd.read_csv('../input/toxicity-of-nitrofen-in-aquatic-systems/nitrofen.csv')

dat.head()
sns.lmplot(x="brood1", y="brood2", hue="conc", data=dat)
f, ax = plt.subplots(figsize=(6, 6))

sns.kdeplot(dat.brood1, dat.brood2, ax=ax)

sns.rugplot(dat.brood1, color="g", ax=ax)

sns.rugplot(dat.brood2, vertical=True, ax=ax);
f, ax = plt.subplots(figsize=(6, 6))

cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)

sns.kdeplot(dat.brood1, dat.brood2, cmap=cmap, n_levels=60, shade=True);
g = sns.jointplot(x="brood1", y="brood2", data=dat, kind="kde", color="m")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)

g.set_axis_labels("$Brood 1$", "$Brood 2$");
f, ax = plt.subplots(figsize=(6, 6))

sns.kdeplot(dat.brood1, dat.brood3, ax=ax)

sns.rugplot(dat.brood1, color="g", ax=ax)

sns.rugplot(dat.brood3, vertical=True, ax=ax);
f, ax = plt.subplots(figsize=(6, 6))

cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)

sns.kdeplot(dat.brood1, dat.brood3, cmap=cmap, n_levels=60, shade=True);
g = sns.jointplot(x="brood1", y="brood3", data=dat, kind="kde", color="m")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)

g.set_axis_labels("$Brood 1$", "$Brood 3$");
f, ax = plt.subplots(figsize=(6, 6))

sns.kdeplot(dat.brood1, dat.total, ax=ax)

sns.rugplot(dat.brood1, color="g", ax=ax)

sns.rugplot(dat.total, vertical=True, ax=ax);
f, ax = plt.subplots(figsize=(6, 6))

cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)

sns.kdeplot(dat.brood1, dat.total, cmap=cmap, n_levels=60, shade=True);
g = sns.jointplot(x="brood1", y="total", data=dat, kind="kde", color="m")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)

g.set_axis_labels("$Brood 1$", "$Total$");