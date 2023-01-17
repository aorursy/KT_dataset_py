from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd
md = pd.read_csv('../input/meningitis_dataset.csv')
md.head()
md.describe()
plt.matshow(md.corr())

plt.colorbar()

plt.show()
sns.lineplot(x='gender_male', y='gender_female', data=md)
p = md.hist(figsize = (20,20))
plt.figure()

sns.distplot(md['gender_male'])

plt.show()

plt.close()
plt.figure()

sns.distplot(md['gender_female'])

plt.show()

plt.close()
sns.kdeplot(data=md['gender_male'],label='Male',shade=True)
sns.kdeplot(data=md['gender_female'],label='Female',shade=True)