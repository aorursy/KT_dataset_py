from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



dat = pd.read_csv('../input/appliances-energy-prediction/KAG_energydata_complete.csv')
dat.head()
dat.describe()
plt.matshow(dat.corr())

plt.colorbar()

plt.show()
p = dat.hist(figsize = (20,20))
sns.regplot(x=dat['T1'], y=dat['T2'])
sns.regplot(x=dat['T1'], y=dat['T3'])
sns.regplot(x=dat['T1'], y=dat['T3'])
sns.regplot(x=dat['T1'], y=dat['T4'])
sns.regplot(x=dat['T1'], y=dat['T5'])
sns.regplot(x=dat['T1'], y=dat['T6'])
sns.regplot(x=dat['T1'], y=dat['T7'])
sns.regplot(x=dat['T1'], y=dat['T8'])
sns.regplot(x=dat['T1'], y=dat['T9'])
sns.regplot(x=dat['T2'], y=dat['T3'])
sns.regplot(x=dat['T2'], y=dat['T4'])
sns.regplot(x=dat['T2'], y=dat['T5'])
sns.regplot(x=dat['Visibility'], y=dat['Windspeed'])