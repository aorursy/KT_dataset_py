from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



dat = pd.read_csv('../input/startups/50_Startups.csv')

dat.head()
sns.lmplot(x="Marketing Spend", y="Profit", hue="State", data=dat)
df = dat.rename({'Marketing Spend': 'ms'}, axis=1)

f, ax = plt.subplots(figsize=(6, 6))

sns.kdeplot(df.ms, df.Profit, ax=ax)

sns.rugplot(df.ms, color="g", ax=ax)

sns.rugplot(df.Profit, vertical=True, ax=ax);
f, ax = plt.subplots(figsize=(6, 6))

cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)

sns.kdeplot(df.ms, df.Profit, cmap=cmap, n_levels=60, shade=True);
g = sns.jointplot(x="ms", y="Profit", data=df, kind="kde", color="m")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)

g.set_axis_labels("$Marketing Spend$", "$Profit$");