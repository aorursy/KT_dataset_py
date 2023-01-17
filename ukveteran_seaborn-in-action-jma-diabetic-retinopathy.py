from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



dat = pd.read_csv('../input/diabetic-retinopathy/diabetic.csv')

dat.head()
sns.lmplot(x="age", y="risk", hue="trt", data=dat);
sns.lmplot(x="age", y="risk", hue="eye", data=dat);
sns.lmplot(x="age", y="risk", hue="laser", data=dat);
sns.lmplot(x="age", y="time", hue="trt", data=dat);
sns.lmplot(x="age", y="time", hue="eye", data=dat);
sns.lmplot(x="age", y="time", hue="laser", data=dat);
f, ax = plt.subplots(figsize=(6, 6))

sns.kdeplot(dat.age, dat.time, ax=ax)

sns.rugplot(dat.age, color="g", ax=ax)

sns.rugplot(dat.time, vertical=True, ax=ax);
g = sns.jointplot(x="age", y="time", data=dat, kind="kde", color="m")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)

g.set_axis_labels("$Age$", "$Time$");
f, ax = plt.subplots(figsize=(6, 6))

cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)

sns.kdeplot(dat.age, dat.time, cmap=cmap, n_levels=60, shade=True);