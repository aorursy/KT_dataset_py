from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



dat = pd.read_csv('../input/isotopic-composition-plutonium-batches/pluton.csv')

dat.head()
sns.lmplot(x="Pu238", y="Pu239",  data=dat);
sns.lmplot(x="Pu238", y="Pu240",  data=dat);
sns.lmplot(x="Pu238", y="Pu241",  data=dat);
f, ax = plt.subplots(figsize=(6, 6))

sns.kdeplot(dat.Pu238, dat.Pu239, ax=ax)

sns.rugplot(dat.Pu238, color="g", ax=ax)

sns.rugplot(dat.Pu239, vertical=True, ax=ax);
f, ax = plt.subplots(figsize=(6, 6))

sns.kdeplot(dat.Pu238, dat.Pu240, ax=ax)

sns.rugplot(dat.Pu238, color="g", ax=ax)

sns.rugplot(dat.Pu240, vertical=True, ax=ax);
f, ax = plt.subplots(figsize=(6, 6))

sns.kdeplot(dat.Pu238, dat.Pu241, ax=ax)

sns.rugplot(dat.Pu238, color="g", ax=ax)

sns.rugplot(dat.Pu241, vertical=True, ax=ax);
f, ax = plt.subplots(figsize=(6, 6))

cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)

sns.kdeplot(dat.Pu238, dat.Pu239, cmap=cmap, n_levels=60, shade=True);
f, ax = plt.subplots(figsize=(6, 6))

cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)

sns.kdeplot(dat.Pu238, dat.Pu240, cmap=cmap, n_levels=60, shade=True);
f, ax = plt.subplots(figsize=(6, 6))

cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)

sns.kdeplot(dat.Pu238, dat.Pu241, cmap=cmap, n_levels=60, shade=True);
g = sns.jointplot(x="Pu238", y="Pu239", data=dat, kind="kde", color="m")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)

g.set_axis_labels("$Pu238", "$Pu239$");
g = sns.jointplot(x="Pu238", y="Pu240", data=dat, kind="kde", color="m")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)

g.set_axis_labels("$Pu238$", "$Pu240$");
g = sns.jointplot(x="Pu238", y="Pu241", data=dat, kind="kde", color="m")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)

g.set_axis_labels("$Pu238$", "$Pu241$");