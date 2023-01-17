from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



dat = pd.read_csv('../input/vertebralcolumndataset/column_3C.csv')

dat.head()
dat.tail()
dat.shape
dat.describe()
dat.info()

dat.columns

dat.dtypes

dat.corr()

dat.plot(subplots=True,figsize=(18,18))

plt.show()
plt.figure(figsize=(15,10))

sns.heatmap(dat.iloc[:,0:15].corr(), annot=True,fmt=".0%")

plt.show()
fig=plt.figure(figsize=(20,15))

ax=fig.gca()

dat.hist(ax=ax)

plt.show()