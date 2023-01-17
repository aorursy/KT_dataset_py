from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



dat = pd.read_csv('../input/data-on-infantmortality/Leinhardt.csv')

dat.head()
sns.lmplot(x="income", y="infant", hue="oil", data=dat)
g = sns.jointplot(x="income", y="infant", data=dat, kind="kde", color="m")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)

g.set_axis_labels("$income$", "$infant$");