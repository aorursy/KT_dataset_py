from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



ed = pd.read_csv('../input/energy_data.csv')
ed.head()
ed.describe()
plt.matshow(ed.corr())

plt.colorbar()

plt.show()
plt.scatter(ed['Community Board'], ed['Census Tract'])
sns.regplot(x=ed['Community Board'], y=ed['Census Tract'])
p = ed.hist(figsize = (20,20))
import matplotlib.pyplot as pl

g = sns.jointplot(x="Community Board", y="Census Tract", data = ed,kind="kde", color="c")

g.plot_joint(pl.scatter, c="w", s=30, linewidth=1, marker="+")