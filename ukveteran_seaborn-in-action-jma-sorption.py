from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



dat = pd.read_csv('../input/sorption-data/sorption.csv')

dat.head()
sns.lmplot(x="m5", y="m10", hue="rep", data=dat)
sns.lmplot(x="m5", y="m10", hue="Cultivar", data=dat)
f, ax = plt.subplots(figsize=(6, 6))

sns.kdeplot(dat.ct, dat.Dose, ax=ax)

sns.rugplot(dat.ct, color="g", ax=ax)

sns.rugplot(dat.Dose, vertical=True, ax=ax);
f, ax = plt.subplots(figsize=(6, 6))

cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)

sns.kdeplot(dat.ct, dat.Dose, cmap=cmap, n_levels=60, shade=True);
g = sns.jointplot(x="ct", y="Dose", data=dat, kind="kde", color="m")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)

g.set_axis_labels("$ct$", "$Dose$");