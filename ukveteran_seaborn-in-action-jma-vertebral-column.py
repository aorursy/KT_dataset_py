from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



dat = pd.read_csv('../input/vertebralcolumndataset/column_2C.csv')

dat.head()
sns.lmplot(x="pelvic_incidence", y="pelvic_tilt", hue="class", data=dat)
sns.lmplot(x="sacral_slope", y="pelvic_radius", hue="class", data=dat)
f, ax = plt.subplots(figsize=(6, 6))

sns.kdeplot(dat.pelvic_incidence, dat.pelvic_tilt, ax=ax)

sns.rugplot(dat.pelvic_incidence, color="g", ax=ax)

sns.rugplot(dat.pelvic_tilt, vertical=True, ax=ax);
f, ax = plt.subplots(figsize=(6, 6))

cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)

sns.kdeplot(dat.pelvic_incidence, dat.pelvic_tilt, cmap=cmap, n_levels=60, shade=True);
g = sns.jointplot(x="sacral_slope", y="pelvic_radius", data=dat, kind="kde", color="m")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)

g.set_axis_labels("$sacral slope$", "$pelvic radius$");