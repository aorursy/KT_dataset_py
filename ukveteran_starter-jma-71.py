from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



dat = pd.read_csv('../input/datasetucimlairquality/AirQualityUCI.csv')
dat.head()
dat.describe()
p = dat.hist(figsize = (20,20))
plt.matshow(dat.corr())

plt.colorbar()

plt.show()
plt.scatter(dat['PT08_S3_Nox'], dat['NO2_GT'])
sns.regplot(x=dat['PT08_S3_Nox'], y=dat['NO2_GT'])
plt.scatter(dat['Nox_GT'], dat['NO2_GT'])
sns.regplot(x=dat['Nox_GT'], y=dat['NO2_GT'])