from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



dat = pd.read_csv('../input/epilepsy-data/epilepsy.csv')

dat.head()
sns.lmplot(x="seizure.rate", y="subject", hue="treatment", data=dat)
df = dat.rename({'seizure.rate': 'seizure'}, axis=1)

f, ax = plt.subplots(figsize=(6, 6))

sns.kdeplot(df.seizure, df.subject, ax=ax)

sns.rugplot(df.seizure, color="g", ax=ax)

sns.rugplot(df.subject, vertical=True, ax=ax);
f, ax = plt.subplots(figsize=(6, 6))

cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)

sns.kdeplot(df.seizure, df.subject, cmap=cmap, n_levels=60, shade=True);
g = sns.jointplot(x="seizure", y="subject", data=df, kind="kde", color="m")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)

g.set_axis_labels("$Seizure Rate$", "$Subject$");