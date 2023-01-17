from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



dat = pd.read_csv('../input/rheumatoid-arthritis-clinical-trial/arthritis.csv')

dat.head()
sns.lmplot(x="baseline", y="time", hue="sex", data=dat)
sns.lmplot(x="baseline", y="time", hue="trt", data=dat)
sns.lmplot(x="baseline", y="time", hue="y", data=dat)