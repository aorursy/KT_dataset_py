from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



dat = pd.read_csv('../input/rollercoaster-tycoon-rides/rollercoasters.csv')

dat.head()
sns.lmplot(x="excitement", y="intensity", hue="intensity_rating", data=dat)
sns.lmplot(x="excitement", y="intensity", hue="excitement_rating", data=dat)
f, ax = plt.subplots(figsize=(6, 6))

sns.kdeplot(dat.excitement, dat.intensity, ax=ax)

sns.rugplot(dat.excitement, color="g", ax=ax)

sns.rugplot(dat.intensity, vertical=True, ax=ax);
f, ax = plt.subplots(figsize=(6, 6))

cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)

sns.kdeplot(dat.excitement, dat.intensity, cmap=cmap, n_levels=60, shade=True);
g = sns.jointplot(x="excitement", y="intensity", data=dat, kind="kde", color="m")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)

g.set_axis_labels("$Excitement$", "$Intensity$");