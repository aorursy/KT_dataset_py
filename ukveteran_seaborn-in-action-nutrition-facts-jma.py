from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import os 
import pandas as pd

dat = pd.read_csv('../input/nutrition-facts/menu.csv')
dat.head()
sns.lmplot(x="Calories", y="Total Fat", hue="Category", data=dat)
g = sns.jointplot(x="Calories", y="Total Fat", data=dat, kind="kde", color="m")
g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("$Calories$", "$Total Fat$");
fig=sns.jointplot(x='Calories',y='Total Fat',kind='hex',data=dat)
sns.jointplot("Calories", "Total Fat", data=dat, kind="kde",space=0)
g = (sns.jointplot("Calories", "Total Fat",data=dat, color="k").plot_joint(sns.kdeplot, zorder=0, n_levels=6))