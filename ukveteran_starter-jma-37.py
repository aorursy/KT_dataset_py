from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



vodi= pd.read_csv('../input/Virat_Kohli_ODI.csv')
vodi.head()
vodi.describe()
plt.matshow(vodi.corr())

plt.colorbar()

plt.show()
sns.lineplot(x='4s', y='6s', data=vodi)
p = vodi.hist(figsize = (20,20))
sns.regplot(vodi['4s'], vodi['6s'])