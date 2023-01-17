from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



dat = pd.read_csv('../input/diabetic-retinopathy/diabetic.csv')

dat.head()
sns.set(rc={'figure.figsize':(19.7,8.27)})

sns.heatmap(dat.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.distplot(dat["time"])
sns.countplot(dat["laser"])
sns.countplot(dat["trt"])
sns.countplot(dat["status"])
sns.countplot(dat["age"])
plt.figure(figsize=(10,6))

sns.catplot(x="age", y="time", data=dat);

plt.ioff()