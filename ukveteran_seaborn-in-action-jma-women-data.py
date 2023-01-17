from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



dat = pd.read_csv('../input/average-heights-and-weights-for-american-women/women.csv')

dat.head()
sns.lmplot(x="height", y="weight", data=dat)
f, ax = plt.subplots(figsize=(6, 6))

sns.kdeplot(dat.height, dat.weight, ax=ax)

sns.rugplot(dat.height, color="g", ax=ax)

sns.rugplot(dat.weight, vertical=True, ax=ax);
f, ax = plt.subplots(figsize=(6, 6))

cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)

sns.kdeplot(dat.height, dat.weight, cmap=cmap, n_levels=60, shade=True);
g = sns.jointplot(x="height", y="weight", data=dat, kind="kde", color="m")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)

g.set_axis_labels("$height$", "$weight$");