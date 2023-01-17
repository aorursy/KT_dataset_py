from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



dat = pd.read_csv('../input/cryptocurrency-market-history-coinmarketcap/all_currencies.csv')

dat.head()
sns.lmplot(x="High", y="Low", hue="Symbol", data=dat)
sns.lmplot(x="Open", y="Close", hue="Symbol", data=dat)
g = sns.jointplot(x="High", y="Low", data=dat, kind="kde", color="m")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)

g.set_axis_labels("$High$", "$Low$");
g = sns.jointplot(x="High", y="Volume", data=dat, kind="kde", color="m")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)

g.set_axis_labels("$High$", "$Volume$");
g = sns.jointplot(x="Low", y="Volume", data=dat, kind="kde", color="m")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)

g.set_axis_labels("$Low$", "$Volume$");