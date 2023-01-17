from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



dat = pd.read_csv('../input/corona-virus-report/novel corona virus situation report who.csv')

dat.head()
sns.lmplot(x="Total", y="China", data=dat);
f, ax = plt.subplots(figsize=(6, 6))

sns.kdeplot(dat.Total, dat.China, ax=ax)

sns.rugplot(dat.Total, color="g", ax=ax)

sns.rugplot(dat.China, vertical=True, ax=ax);
g = sns.jointplot(x="Total", y="China", data=dat, kind="kde", color="m")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)