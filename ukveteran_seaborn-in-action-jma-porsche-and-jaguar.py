from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



dat = pd.read_csv('../input/porsche-and-jaguar-prices/PorscheJaguar.csv')

dat.head()
sns.lmplot(x="Price", y="Mileage", hue="Car", data=dat)
sns.lmplot(x="Price", y="Age", hue="Car", data=dat)
f, ax = plt.subplots(figsize=(6, 6))

sns.kdeplot(dat.Price, dat.Mileage, ax=ax)

sns.rugplot(dat.Price, color="g", ax=ax)

sns.rugplot(dat.Mileage, vertical=True, ax=ax);
f, ax = plt.subplots(figsize=(6, 6))

cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)

sns.kdeplot(dat.Price, dat.Mileage, cmap=cmap, n_levels=60, shade=True);
g = sns.jointplot(x="Price", y="Mileage", data=dat, kind="kde", color="m")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)

g.set_axis_labels("$Price$", "$Mileage$");
f, ax = plt.subplots(figsize=(6, 6))

sns.kdeplot(dat.Price, dat.Age, ax=ax)

sns.rugplot(dat.Price, color="g", ax=ax)

sns.rugplot(dat.Age, vertical=True, ax=ax);
f, ax = plt.subplots(figsize=(6, 6))

cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)

sns.kdeplot(dat.Price, dat.Age, cmap=cmap, n_levels=60, shade=True);
g = sns.jointplot(x="Price", y="Age", data=dat, kind="kde", color="m")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)

g.set_axis_labels("$Price$", "$Age$");
f, ax = plt.subplots(figsize=(6, 6))

sns.kdeplot(dat.Mileage, dat.Age, ax=ax)

sns.rugplot(dat.Mileage, color="g", ax=ax)

sns.rugplot(dat.Age, vertical=True, ax=ax);
f, ax = plt.subplots(figsize=(6, 6))

cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)

sns.kdeplot(dat.Mileage, dat.Age, cmap=cmap, n_levels=60, shade=True);
g = sns.jointplot(x="Mileage", y="Age", data=dat, kind="kde", color="m")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)

g.set_axis_labels("$Mileage$", "$Age$");