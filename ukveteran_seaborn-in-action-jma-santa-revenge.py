from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



dat = pd.read_csv('../input/santa-2019-revenge-of-the-accountants-10-days/family_data.csv')

dat.head()
sns.lmplot(x="choice_0", y="choice_1", data=dat);
sns.lmplot(x="choice_0", y="choice_2", data=dat);
sns.lmplot(x="choice_0", y="choice_3", data=dat);
sns.lmplot(x="choice_0", y="choice_4", data=dat);
sns.lmplot(x="choice_0", y="choice_5", data=dat);
sns.lmplot(x="choice_0", y="choice_6", data=dat);
sns.lmplot(x="choice_0", y="choice_8", data=dat);
sns.lmplot(x="choice_0", y="choice_9", data=dat);
f, ax = plt.subplots(figsize=(6, 6))

sns.kdeplot(dat.choice_0, dat.choice_1, ax=ax)

sns.rugplot(dat.choice_0, color="g", ax=ax)

sns.rugplot(dat.choice_1, vertical=True, ax=ax);
f, ax = plt.subplots(figsize=(6, 6))

sns.kdeplot(dat.choice_0, dat.choice_2, ax=ax)

sns.rugplot(dat.choice_0, color="g", ax=ax)

sns.rugplot(dat.choice_2, vertical=True, ax=ax);
f, ax = plt.subplots(figsize=(6, 6))

sns.kdeplot(dat.choice_0, dat.choice_3, ax=ax)

sns.rugplot(dat.choice_0, color="g", ax=ax)

sns.rugplot(dat.choice_3, vertical=True, ax=ax);
f, ax = plt.subplots(figsize=(6, 6))

sns.kdeplot(dat.choice_0, dat.choice_4, ax=ax)

sns.rugplot(dat.choice_0, color="g", ax=ax)

sns.rugplot(dat.choice_4, vertical=True, ax=ax);
f, ax = plt.subplots(figsize=(6, 6))

sns.kdeplot(dat.choice_0, dat.choice_5, ax=ax)

sns.rugplot(dat.choice_0, color="g", ax=ax)

sns.rugplot(dat.choice_5, vertical=True, ax=ax);
f, ax = plt.subplots(figsize=(6, 6))

sns.kdeplot(dat.choice_0, dat.choice_6, ax=ax)

sns.rugplot(dat.choice_0, color="g", ax=ax)

sns.rugplot(dat.choice_6, vertical=True, ax=ax);
f, ax = plt.subplots(figsize=(6, 6))

sns.kdeplot(dat.choice_0, dat.choice_7, ax=ax)

sns.rugplot(dat.choice_0, color="g", ax=ax)

sns.rugplot(dat.choice_7, vertical=True, ax=ax);
f, ax = plt.subplots(figsize=(6, 6))

sns.kdeplot(dat.choice_0, dat.choice_8, ax=ax)

sns.rugplot(dat.choice_0, color="g", ax=ax)

sns.rugplot(dat.choice_8, vertical=True, ax=ax);
f, ax = plt.subplots(figsize=(6, 6))

sns.kdeplot(dat.choice_0, dat.choice_9, ax=ax)

sns.rugplot(dat.choice_0, color="g", ax=ax)

sns.rugplot(dat.choice_9, vertical=True, ax=ax);
g = sns.jointplot(x="choice_0", y="choice_1", data=dat, kind="kde", color="m")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)
g = sns.jointplot(x="choice_0", y="choice_2", data=dat, kind="kde", color="m")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)
g = sns.jointplot(x="choice_0", y="choice_3", data=dat, kind="kde", color="m")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)
g = sns.jointplot(x="choice_0", y="choice_4", data=dat, kind="kde", color="m")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)
g = sns.jointplot(x="choice_0", y="choice_5", data=dat, kind="kde", color="m")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)
g = sns.jointplot(x="choice_0", y="choice_6", data=dat, kind="kde", color="m")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)
g = sns.jointplot(x="choice_0", y="choice_7", data=dat, kind="kde", color="m")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)
g = sns.jointplot(x="choice_0", y="choice_8", data=dat, kind="kde", color="m")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)
g = sns.jointplot(x="choice_0", y="choice_9", data=dat, kind="kde", color="m")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)