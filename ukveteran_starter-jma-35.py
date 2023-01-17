from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



pat= pd.read_csv('../input/patients.csv')
pat.head()
pat.describe()
plt.matshow(pat.corr())

plt.colorbar()

plt.show()
plt.scatter(pat['Systolic'], pat['Weight'])
sns.regplot(pat['Systolic'], pat['Weight'])
sns.lineplot(x='Systolic', y='Weight', data=pat)
plt.style.use('fast')

sns.jointplot(x='Systolic', y='Weight', data=pat)

plt.show()
q = sns.boxenplot(pat['Systolic'], pat['Weight'], palette = 'rocket')