from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



dat = pd.read_csv('../input/heart-disease-dataset-from-uci/HeartDisease.csv')

dat.head()
sns.lmplot(x="trestbps", y="thalach", hue="Sex", data=dat)
sns.lmplot(x="trestbps", y="thalach", hue="Place", data=dat)
g = sns.jointplot(x="Age", y="trestbps", data=dat, kind="kde", color="m")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)

g.set_axis_labels("$Age$", "$Trestbps$");
g = sns.jointplot(x="Age", y="thalach", data=dat, kind="kde", color="m")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)

g.set_axis_labels("$Age$", "$Thalach$");
g = sns.jointplot(x="trestbps", y="thalach", data=dat, kind="kde", color="m")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)

g.set_axis_labels("$Trestbps$", "$Thalach$");