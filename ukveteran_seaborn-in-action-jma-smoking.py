from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



dat = pd.read_csv('../input/smoking/smoking.csv')

dat.head()
sns.lmplot(x="sbp", y="age", hue="male", data=dat)
sns.lmplot(x="sbp", y="age", hue="smoker", data=dat)