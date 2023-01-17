from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



sol= pd.read_csv('../input/Set for Life.csv')
sol.head()
sol.describe()
plt.matshow(sol.corr())

plt.colorbar()

plt.show()
plt.scatter(sol['Ball 1'], sol['Life Ball'])
sns.regplot(x=sol['Ball 1'], y=sol['Life Ball'])
sns.lineplot(x='Ball 1', y='Life Ball', data=sol)
plt.style.use('fast')

sns.jointplot(x = 'Ball 1', y = 'Life Ball', data = sol)

plt.show()
q = sns.boxenplot(x = sol['Ball 1'], y = sol['Life Ball'], palette = 'rocket')