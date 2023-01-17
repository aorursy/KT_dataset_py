from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



dat = pd.read_csv('../input/fertility-and-contraception/Robey.csv')

dat.head()
sns.distplot(dat["tfr"])
sns.distplot(dat["contraceptors"])
sns.scatterplot(x='tfr',y='contraceptors',data=dat)
sns.countplot(dat["tfr"])
sns.countplot(dat["contraceptors"])
plt.figure(figsize=(10,6))

sns.catplot(x="tfr", y="contraceptors", data=dat);

plt.ioff()